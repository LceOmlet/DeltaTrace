"""Read the unchanged final FA gate; never supply a finite coefficient.

All model/FA calls and the original DT result remain owned by the existing
runner. The extra projection and sigmoid evaluations are charged diagnostic
readouts. The decomposition is an arithmetic error ledger, not a correction,
a causal share, a new tolerance, or evidence of improved attribution quality.
"""
from contextlib import contextmanager
import inspect
import os
import time

import torch

from passive_suboperations import PassiveSuboperations


class PassiveAttentionGate(PassiveSuboperations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.layers == [31] or self.layers == (31,)
        self.gate_bank = {}
        self.gate_owner = self.runner.boundaries.attention_gate
        self.globals = self.runner.attribute.__func__.__globals__
        self.attention_owner = self.globals['attention_finite_pullback']
        self.extra_projection_calls = 0
        self.extra_sigmoid_calls = 0
        self.gate_readout_seconds = 0.
        self.read_input = os.environ.get('DT_ATTENTION_CORE_INPUT') == '1'
        self.input_owner = self.runner.boundaries.attention_input

    def __enter__(self):
        super().__enter__()
        module = self.text.layers[31].self_attn
        def qproj(_module, _args, output):
            if self.mode == 'DT':
                g = output.reshape(*output.shape[:2], module.config.num_attention_heads,
                                   2*module.head_dim)[..., module.head_dim:]
                self.gate_bank['g'], self.gate_bank['s'] = self.gate_endpoints(g)
        def product(_module, args):
            if self.mode == 'DT':
                self.gate_bank['p'] = self.cpu(args[0]).reshape(
                    *args[0].shape[:2], module.config.num_attention_heads, module.head_dim)
        self.handles.append(module.q_proj.register_forward_hook(qproj))
        self.handles.append(module.o_proj.register_forward_pre_hook(product))
        self.runner.boundaries.attention_gate = self.gate
        self.globals['attention_finite_pullback'] = self.attention
        if self.read_input:
            self.runner.boundaries.attention_input = self.input
        return self

    def __exit__(self, *args):
        self.runner.boundaries.attention_gate = self.gate_owner
        self.globals['attention_finite_pullback'] = self.attention_owner
        self.runner.boundaries.attention_input = self.input_owner
        return super().__exit__(*args)

    def gate_endpoints(self, g):
        # Re-evaluate the same native Torch sigmoid on its actual CUDA operand,
        # in small slices. The native product comparison below retains any
        # discrepancy rather than assuming a CPU BF16 sigmoid is equivalent.
        saved_g = self.cpu(g)
        saved_s = torch.empty_like(saved_g)
        for start in range(0, g.shape[1], 128):
            saved_s[:, start:start+128] = self.cpu(g[:, start:start+128].sigmoid())
            self.extra_sigmoid_calls += 1
        return saved_g, saved_s

    def attention(self, *args, **kwargs):
        if self.current_layer == 31:
            assert kwargs.get('pv_rule', 'content1') == 'content1'
            self.gate_bank['o'] = self.cpu(args[1]['attention_output'])
            if self.read_input:
                for name in ('query', 'key', 'value'):
                    self.gate_bank[name] = self.cpu(args[1][name])
        return self.attention_owner(*args, **kwargs)

    def input(self, *args, **kwargs):
        value = self.input_owner(*args, **kwargs)
        if self.current_layer == 31:
            for name, operand in zip(('dq', 'dk', 'dv'), args[:3]):
                self.gate_bank[name] = self.cpu(operand)
        return value

    def gate(self, *args, **kwargs):
        value = self.gate_owner(*args, **kwargs)
        if self.current_layer == 31:
            from qwen35_decoder_finite import _linear_transpose
            begin = time.perf_counter()
            q0, q1, o0, upstream, weight, heads, dimension = args
            g0 = q0.reshape(*q0.shape[:2], heads, 2*dimension)[..., dimension:]
            g1 = q1.reshape(*q1.shape[:2], heads, 2*dimension)[..., dimension:]
            assert torch.equal(self.cpu(g0), self.gate_bank['g'][0::2])
            assert torch.equal(self.cpu(g1), self.gate_bank['g'][1::2])
            assert torch.equal(self.cpu(o0), self.gate_bank['o'][0::2])
            self.gate_bank['mc'] = self.cpu(value[0].transpose(1, 2))
            self.gate_bank['mg'] = self.cpu(value[1])
            projected = torch.empty((*upstream.shape[:2], heads, dimension), dtype=torch.float32)
            for first in range(0, upstream.shape[1], 128):
                piece = _linear_transpose(upstream[:, first:first+128], weight)
                projected[:, first:first+128] = self.cpu(piece).reshape(
                    *piece.shape[:2], heads, dimension)
                self.extra_projection_calls += 1
                del piece
            self.gate_bank['mp'] = projected
            self.gate_bank['mup'] = self.cpu(upstream)
            self.linear_readout_owner = inspect.getsourcefile(_linear_transpose)
            self.gate_readout_seconds += time.perf_counter()-begin
        return value

    @contextmanager
    def native_scope(self):
        factory = self.globals['NativeDenseAttentionCapture']
        observer = self
        class ReadGate(factory):
            def retain(self, name, value):
                if name == 'q_proj_output':
                    module = self.module
                    g = value.reshape(*value.shape[:2], module.config.num_attention_heads,
                                      2*module.head_dim)[..., module.head_dim:]
                    self.values['g'], self.values['s'] = observer.gate_endpoints(g)
                elif name in ('attention_output', 'o_proj_input', 'output') or (
                        observer.read_input and name in ('query', 'key', 'value')):
                    super().retain(name, value)
        capture = self.native_capture_type(ReadGate)(self.text.layers[31].self_attn, self.globals['flash_attention_forward'],
            self.globals['flash_attn_varlen_func'], self.globals['flash_attn_func'],
            destination='cpu', copy_tensors=True,
            retained_names={'q_proj_output', 'attention_output', 'o_proj_input', 'output'} |
                ({'query', 'key', 'value'} if self.read_input else set()) | self.extra_native_names())
        with capture:
            yield
        self.native_gate = capture.values
        self.native_capture_calls = capture.calls
        self.native_capture_finished(capture)

    def native_capture_type(self, capture_type):
        return capture_type

    def extra_native_names(self):
        return set()

    def native_capture_finished(self, capture):
        pass

    def finish_point(self, point, data):
        super().finish_point(point, data)
        row = next(i for i, q in enumerate(self.active['queries'])
                   if q is not None and self.rows[i]['row']['traj_uid'] == point['traj_uid'])
        start = self.starts[row]
        length = self.rows[row]['selected'].numel()-start
        bank, native = self.gate_bank, self.native_gate
        names = ('coefficient_recompute', 'product_background', 'sigmoid_rounded_secant',
                 'sigmoid_background', 'native_sigmoid_rounding', 'native_product_rounding')
        totals = {name: 0. for name in names}
        totals.update(gate_residual=0., core_and_input_residual=0., projection_residual=0.,
                      native_product_recompute_maxabs=0., gate_coefficient=0., content_coefficient=0.)
        smooth_exceeded, finite_pairs = 0, 0
        tick = time.perf_counter()
        for first in range(0, length, 128):
            last = min(first+128, length)
            def d(name, factual=True):
                return bank[name][2*row+int(factual), first:last].double()
            def n(name, factual=False):
                return native[name][2*row+int(factual), start+first:start+last].double()
            oF, oR, oD = d('o'), d('o', False), n('attention_output')
            gF, gR, gD = d('g'), d('g', False), n('g')
            sF, sR, sD = d('s'), d('s', False), n('s')
            pF = d('p')
            pD = n('o_proj_input').reshape_as(pF)
            mc, mg, mp = (bank[k][row, first:last].double() for k in ('mc','mg','mp'))
            dg, do, ds = gF-gD, oF-oD, sF-sD
            joint = gF-gR
            safe = torch.where(joint != 0, joint, torch.ones_like(joint))
            smoothF, smoothR, smoothD = gF.sigmoid(), gR.sigmoid(), gD.sigmoid()
            derivativeR = smoothR*(1-smoothR)
            cBF = torch.where(joint != 0, (sF-sR)/safe, derivativeR)
            cSmooth = torch.where(joint != 0, (smoothF-smoothR)/safe, derivativeR)
            deltaSmooth = smoothF-smoothD
            terms = dict(
                coefficient_recompute=(mc-mp*sF)*do+(mg-mp*oR*cBF)*dg,
                product_background=mp*(oR-oD)*ds,
                sigmoid_rounded_secant=mp*oR*(cBF-cSmooth)*dg,
                sigmoid_background=mp*oR*(cSmooth*dg-deltaSmooth),
                native_sigmoid_rounding=mp*oR*(deltaSmooth-ds),
                native_product_rounding=mp*(oF*sF-oD*sD-(pF-pD)))
            for name, value in terms.items():
                totals[name] += float(value.sum())
            totals['content_coefficient'] += float((mc*do).sum())
            totals['gate_coefficient'] += float((mg*dg).sum())
            totals['gate_residual'] += float((mc*do+mg*dg-mp*(pF-pD)).sum())
            # Native product validation is an observation, not a new tolerance.
            for endpoint in (0, 1):
                raw_o = native['attention_output'][2*row+endpoint, start+first:start+last]
                raw_s = native['s'][2*row+endpoint, start+first:start+last]
                raw_p = native['o_proj_input'][2*row+endpoint, start+first:start+last].reshape_as(raw_o)
                totals['native_product_recompute_maxabs'] = max(totals['native_product_recompute_maxabs'],
                    float((raw_o*raw_s-raw_p).double().abs().max()))
            finite_pairs += int((joint != 0).sum())
            smooth_exceeded += int(((joint != 0) & (cBF.abs() > .25)).sum())
        # The old matched output contraction keeps the projection's native
        # floating-point behavior separate from the gate product ledger.
        mh_a = data['31']['mh_a']['matched']
        product_effect = totals['content_coefficient']+totals['gate_coefficient']-totals['gate_residual']
        totals['projection_residual'] = product_effect-mh_a
        totals['core_and_input_residual'] = data['31']['mn1_n1']['matched']-totals['content_coefficient']-totals['gate_coefficient']
        totals['gate_decomposition_roundoff'] = sum(totals[name] for name in names)-totals['gate_residual']
        totals['whole_mixer_decomposition_roundoff'] = totals['core_and_input_residual']+totals['gate_residual']+totals['projection_residual']-data['31']['matched_terms']['mixer']
        totals.update(nonzero_joint_gate_pairs=finite_pairs, BF16_secant_above_smooth_sigmoid_Lipschitz=smooth_exceeded,
            comparison='The .25 bound belongs to the smooth sigmoid, not an official BF16 tolerance. '
                'Quantized secants may exceed it; counts alone do not establish attribution error or a repair.',
            native_capture_calls=self.native_capture_calls)
        point['final_FA_gate_ledger'] = totals
        if self.read_input:
            contractions = {}
            shapes = {}
            for operand, coefficient in (('query', 'dq'), ('key', 'dk'), ('value', 'dv')):
                m, factual, deleted = bank[coefficient], bank[operand], native[operand]
                shapes[operand] = dict(coefficient=list(m.shape), DT_endpoint=list(factual.shape),
                                       native_endpoint=list(deleted.shape))
                assert m.shape[1] % deleted.shape[1] == 0
                assert factual.shape[1] == deleted.shape[1]
                total = 0.
                for first in range(0, length, 128):
                    last = min(first+128, length)
                    # Cached Q is suffix-local; compact K/V use original
                    # context coordinates. The actual owner provides both.
                    offset = first if operand == 'query' else start+first
                    f = factual[2*row+1, :, offset:offset+last-first].double()
                    d = deleted[2*row, :, start+first:start+last].double()
                    c = m[row, :, first:last].double()
                    c = c.reshape(deleted.shape[1], -1, last-first, c.shape[-1]).sum(1)
                    assert f.shape == d.shape == c.shape
                    total += float((c*(f-d)).sum())
                contractions[operand] = total
            qkv = sum(contractions.values())
            input_error = data['31']['mn1_n1']['matched']-totals['gate_coefficient']-qkv
            FA_error = qkv-totals['content_coefficient']
            point['final_FA_input_ledger'] = dict(operand_shapes=shapes, contractions=contractions,
                QKV_total=qkv, input_projection_norm_RoPE_residual=input_error,
                finite_FA_core_residual=FA_error,
                decomposition_roundoff=input_error+FA_error-totals['core_and_input_residual'],
                interpretation='Actual original dq/dk/dv contractions; GQA heads are reduced for the observed native operand. '
                    'Input residual includes original projection, QK normalization, RoPE and their rounding. '
                    'Core residual includes finite FA propagation and native attention arithmetic. '
                    'Neither contraction is a log probability or an isolated kernel error. No coefficient is replaced.')
        self.gate_readout_seconds += time.perf_counter()-tick

    def bytes(self):
        return super().bytes()+sum(v.numel()*v.element_size() for v in self.gate_bank.values())
