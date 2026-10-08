"""Passive contractions of existing DT callbacks and native module outputs.

This module never supplies a coefficient, endpoint or credit to the owner.
Residual additions are measured separately: BF16 addition is not assumed to
be an exact real-number identity. Recomputed head coefficients are diagnostic
only, and their difference from the actual compiled coefficient is retained.
"""
import copy
import time
import torch


class PassiveSuboperations:
    def __init__(self, runner, layers, effect, bank, active, starts, rows, selection):
        self.runner, self.layers, self.effect, self.bank = runner, layers, effect, bank
        self.active, self.starts, self.rows, self.selection = active, starts, rows, selection
        self.handles = []
        self.facts, self.coefficients, self.pending = {}, {}, {}
        self.head = {}
        self.mode, self.current_layer = 'DT', None
        self.text = runner.model.model.language_model
        self.norm_owner = runner.boundaries.norm_residual
        self.answer_owner = runner.answer
        self.norm_calls = 0
        self.readout_seconds = 0.0

    @staticmethod
    def cpu(value):
        return value.detach().to('cpu', copy=True)

    def __enter__(self):
        for i in self.layers:
            layer = self.text.layers[i]
            modules = dict(n1=layer.input_layernorm,
                           n2=layer.post_attention_layernorm,
                           a=layer.self_attn if layer.block_type == 'full_attention' else layer.linear_attn,
                           u=layer.mlp)
            for name, module in modules.items():
                def hook(_module, args, output, i=i, name=name):
                    value = output[0] if isinstance(output, tuple) else output
                    if self.mode == 'DT':
                        self.pending[i, name] = self.cpu(value[1::2])
                    else:
                        self.contract(i, name, value)
                    if name == 'n2':
                        if self.mode == 'DT':
                            self.pending[i, 'h'] = self.cpu(args[0][1::2])
                        else:
                            self.contract(i, 'h', args[0])
                self.handles.append(module.register_forward_hook(hook))
            def before(_module, args, kwargs, i=i):
                if self.mode == 'native':
                    self.contract(i, 'x', args[0] if args else kwargs['hidden_states'])
            self.handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        def norm_output(_module, _args, output):
            if self.mode == 'DT':
                self.pending['final_norm'] = self.cpu(output)
            else:
                self.head['native_norm'] = self.cpu(self.selection.pack_hidden(output))
        self.handles.append(self.text.norm.register_forward_hook(norm_output))
        self.runner.boundaries.norm_residual = self.norm
        self.runner.answer = self.answer
        return self

    def __exit__(self, *_exc):
        self.runner.boundaries.norm_residual = self.norm_owner
        self.runner.answer = self.answer_owner
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def enter_decoder(self, index):
        self.current_layer = index if index in self.layers else None
        self.norm_calls = 0

    def leave_decoder(self):
        self.current_layer = None

    def norm(self, *args, **kwargs):
        value = self.norm_owner(*args, **kwargs)
        i = self.current_layer
        if i is not None:
            self.norm_calls += 1
            if self.norm_calls == 1:
                self.coefficients[i, 'n2'] = self.cpu(args[3])
                self.coefficients[i, 'h'] = self.cpu(value)
                for name in ('n1', 'n2', 'a', 'u', 'h'):
                    self.facts[i, name] = self.pending.pop((i, name))
            else:
                assert self.norm_calls == 2
                self.coefficients[i, 'n1'] = self.cpu(args[3])
        return value

    def answer(self, z, head, selected, *args, **kwargs):
        from compiled_logprob_seed import seed_with_checks
        from qwen35_decoder_finite import _linear_transpose
        value = self.answer_owner(z, head, selected, *args, **kwargs)
        tick = time.perf_counter()
        assert selected.outcome_token_ids is None and not args
        cpu_selection = copy.copy(selected)
        cpu_selection.paired_samples = selected.paired_samples.cpu()
        cpu_selection.paired_positions = selected.paired_positions.cpu()
        self.head['norm_factual'] = cpu_selection.pack_hidden(self.pending.pop('final_norm'))[1::2]
        self.head['actual_coefficient'] = self.cpu(value[1]['packed_hidden'])
        self.head['samples'] = selected.samples.cpu()
        seeds, projected, factual_logits = [], [], []
        for first in range(0, len(selected.labels), 128):
            last = first + 128
            z0, z1 = z[2*first:2*last:2].float(), z[2*first+1:2*last:2].float()
            seed, valid = seed_with_checks(z0, z1, selected.labels[first:last])
            assert bool(valid)
            projected.append(self.cpu(_linear_transpose(seed, head.weight)))
            seeds.append(self.cpu(seed))
            factual_logits.append(self.cpu(z[2*first+1:2*last:2]))
            del seed, z0, z1
        self.head['seed'] = torch.cat(seeds)
        self.head['recomputed_coefficient'] = torch.cat(projected)
        self.head['factual_logits'] = torch.cat(factual_logits)
        self.readout_seconds += time.perf_counter() - tick
        return value

    def contraction(self, coefficient, factual, native, row, *, packed=False):
        if packed:
            m, f, pair = coefficient, factual, native
            m, f = m.to(pair.device), f.to(pair.device)
            # The same owner bounds its FP64 contraction temporaries.
            pair = torch.stack((pair[0::2], pair[1::2])).contiguous()
        else:
            start = self.starts[row]
            length = self.rows[row]['selected'].numel() - start
            m = coefficient[row:row+1, :length].to(native.device)
            f = factual[row:row+1, :length].to(native.device)
            pair = native[2*row:2*row+2, start:start+length]
        if packed:
            m, f = m.unsqueeze(0), f.unsqueeze(0)
        native_value = float(self.effect(m, pair).sum())
        matched = torch.cat((pair[0:1], f), dim=0)
        matched_value = float(self.effect(m, matched).sum())
        return dict(native=native_value, matched=matched_value)

    def contract(self, i, name, output):
        specs = {'x':(('mx_x', self.bank[i][0], self.bank[i][1]),
                       ('mh_x', self.coefficients[i,'h'], self.bank[i][1])),
                 'h':(('mh_h', self.coefficients[i,'h'], self.facts[i,'h']),
                       ('my_h', self.bank[i+1][0], self.facts[i,'h'])),
                 'n1':(('mn1_n1', self.coefficients[i,'n1'], self.facts[i,'n1']),),
                 'n2':(('mn2_n2', self.coefficients[i,'n2'], self.facts[i,'n2']),),
                 'a':(('mh_a', self.coefficients[i,'h'], self.facts[i,'a']),),
                 'u':(('my_u', self.bank[i+1][0], self.facts[i,'u']),)}
        for row, query in enumerate(self.active['queries']):
            if query is None:
                continue
            dest = self.active['suboperations'][row].setdefault(str(i), {})
            for label, coefficient, factual in specs[name]:
                dest[label] = self.contraction(coefficient, factual, output, row)

    def head_contractions(self, logits):
        tick = time.perf_counter()
        for row, query in enumerate(self.active['queries']):
            if query is None:
                continue
            mask = self.head['samples'] == row
            selected = torch.where(mask)[0]
            norm = self.head['native_norm'].view(-1, 2, self.head['native_norm'].shape[-1])[mask].flatten(0, 1).to(logits.device)
            dest = self.active['suboperations'][row]['head'] = {}
            for key, name in (('actual_coefficient','actual_norm'), ('recomputed_coefficient','recomputed_norm')):
                dest[name] = self.contraction(self.head[key][mask], self.head['norm_factual'][mask], norm, row, packed=True)
            sums = dict(native=0.0, matched=0.0)
            for first in range(0, len(selected), 128):
                indices = selected[first:first+128]
                paired = (2*indices[:,None]+torch.arange(2)[None,:]).flatten().to(logits.device)
                point = self.contraction(self.head['seed'][indices], self.head['factual_logits'][indices],
                    logits.index_select(0,paired), row, packed=True)
                for key in sums:
                    sums[key] += point[key]
            dest['logits'] = sums
        self.head.pop('native_norm')
        self.readout_seconds += time.perf_counter() - tick

    def bytes(self):
        return sum(v.numel()*v.element_size() for d in (self.facts,self.coefficients,self.pending,self.head)
                   for v in d.values() if isinstance(v,torch.Tensor))

    def finish_point(self, point, data):
        for i in self.layers:
            dest = data[str(i)]
            upper = point['boundaries'][str(i+1)]
            dest['my_y'] = dict(native=upper['value'], matched=upper['DT_factual_minus_native_deleted_effect'])
            for mode in ('native','matched'):
                e = {k:v[mode] for k,v in dest.items() if isinstance(v,dict) and mode in v}
                terms = dict(norm1=e['mx_x']-e['mh_x']-e['mn1_n1'],
                    mixer=e['mn1_n1']-e['mh_a'],
                    residual_add1=e['mh_x']+e['mh_a']-e['mh_h'],
                    norm2=e['mh_h']-e['my_h']-e['mn2_n2'],
                    MLP=e['mn2_n2']-e['my_u'],
                    residual_add2=e['my_h']+e['my_u']-e['my_y'])
                dest[mode+'_terms'] = terms
                dest[mode+'_telescoping_roundoff'] = sum(terms.values())-(e['mx_x']-e['my_y'])
        head = data['head']
        pre = point['boundaries']['32']
        for mode in ('native','matched'):
            p = pre['value' if mode=='native' else 'DT_factual_minus_native_deleted_effect']
            a,b,c = (head[name][mode] for name in ('actual_norm','recomputed_norm','logits'))
            delta = point['native_single_d']+(point['DT_minus_native_factual_score'] if mode=='matched' else 0)
            terms = dict(final_norm=p-a, seed_projection_recompute=a-b,
                         linear_projection=b-c, logsoftmax_background=c-delta)
            head[mode+'_terms'] = terms
            head[mode+'_telescoping_roundoff'] = sum(terms.values())-(p-delta)
        point['suboperations'] = data
