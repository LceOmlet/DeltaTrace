"""Account for fixed-set partial-deletion mismatch through unchanged native DT."""
import argparse
import gc
import hashlib
import inspect
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'environment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--phase', choices=['pilot', 'development16'], default='pilot')
    a = p.parse_args()
    assert a.phase == 'pilot'
    refs = [('/tmp/codex_clean_development16_20260909_v1/qwen35/results.json',
             '04c59aaee006b49bb1c93d5ded737805b17570c3517aa045370bd38d37226cdb'),
            ('/tmp/codex_clean_development16_20260909_mh_recovery_v1/qwen35/results.json',
             '5999968354f16fe3760a4e2f9bfae9401524d366648cfe7f11fe563339719467')]
    rows = []
    for path, digest in refs:
        raw = Path(path).read_bytes(); assert sha(raw) == digest
        rows.extend(r for r in json.loads(raw)['cases'] if r['status'] == 'complete')
    rows = sorted([r for r in rows if a.phase != 'pilot' or r['index'] < 2], key=lambda r: (r['dataset'], r['index']))
    assert len(rows) == (4 if a.phase == 'pilot' else 16)
    env = json.loads(a.environment.read_bytes())['qwen35']
    os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0')
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/deltatrace_clean_v1_inductor')
    for d in env.get('dependency_overlays', []): sys.path.insert(0, d)
    for d in ('deltatrace/clean/qwen35', 'deltatrace/accelerated'): sys.path.insert(0, str(a.release / d))
    import numpy as np
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from deferred import make_deferred_qwen35
    from qwen35_answer_finite import PackedAnswerTargets
    from native_target_logit_rows import NativeTargetLogitRows
    from finite_fla_gpu import verify_native_sources
    from qwen35_decoder_finite import _linear_transpose, _partial_rotation_transpose
    from qwen35_gdn_finite import _l2_pullback
    from signed_secant_rules import rmsnorm_secant_pullback
    from qwen35_core_order_trace_controller import Qwen35DenseFiniteRunner as TraceRunner
    from transformers.integrations.flash_attention import flash_attention_forward
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256, RightPaddedLengths
    torch.set_num_threads(4); torch.manual_seed(73); torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(128, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit = max(512, torch._dynamo.config.accumulated_cache_size_limit)
    a.output.mkdir(exist_ok=False)
    report = {'status': 'loading', 'script_sha256': sha(Path(__file__).read_bytes()), 'phase': a.phase,
              'references': refs, 'cases': [], 'calls': [], 'generation_calls': 0, 'FT_calls': 0, 'metric_calls': 0}
    vectors = {}

    def save():
        q = a.output / 'results.partial'; q.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); q.replace(a.output / 'results.json')

    def timed(name, fn):
        report['status'] = name; save(); torch.cuda.synchronize(); tick = time.perf_counter()
        value = fn(); torch.cuda.synchronize()
        report['calls'].append({'name': name, 'seconds': time.perf_counter() - tick}); save(); return value

    try:
        model = timed('load', lambda: Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'], dtype=torch.bfloat16,
                      attn_implementation='eager', device_map={'': 'cuda:0'}, local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        identity = {n: (type(m).forward, m.forward) for n, m in model.named_modules()}
        with torch.no_grad():
            initial = torch.tensor(rows[0]['input_ids'], device='cuda')[None]
            timed('native_initialization', lambda: model(input_ids=initial, attention_mask=torch.ones_like(initial), use_cache=False))
            del initial
        model.set_attn_implementation('flash_attention_2'); verify_native_sources(env['native_stage_source_sha256'])
        fa = VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256'])
        runner, report['baseline_sources'] = make_deferred_qwen35(a.release, model, fa, None)
        trace_runner=TraceRunner.__new__(TraceRunner)
        trace_runner.__dict__.update(runner.__dict__)
        assert trace_runner.__dict__==runner.__dict__
        derivation=json.loads((Path(__file__).parent/'qwen35_core_order_trace_derivation.json').read_bytes())
        assert sha((a.release/derivation['controller']['original_path']).read_bytes())==derivation['controller']['original_sha256']
        assert sha((Path(__file__).parent/'qwen35_core_order_trace_controller.py').read_bytes())==derivation['controller']['derived_sha256']
        report['observer_derivation']=derivation
        layers = model.model.language_model.layers
        norm = model.model.language_model.norm
        for row in rows:
            key = f"{row['dataset']}_{row['index']}"; n = len(row['input_ids'])
            ids = torch.tensor(row['input_ids'], device='cuda')[None]
            assert sha(ids.cpu().numpy().tobytes()) == row['input_sha256']
            pair = ids.repeat(2, 1); pair[0, [row['user_positions'][j] for j in row['keep']]] = int(ids[0, -1])
            mask = torch.ones_like(pair)
            selection = PackedAnswerTargets([{'target_ids': ids[0, row['prompt_length']:].cpu(), 'prompt_length': row['prompt_length']}],
                                            [list(range(row['target_length']))], n, 'cuda')
            selector = NativeTargetLogitRows(selection)
            plain, plain_detail = timed(key + '/plain', lambda: runner.attribute(pair, mask, selection))
            vectors[key + '/plain'] = plain.numpy()
            partials = {}; partial_logprobs = {}; deleted_sets = {}; input_hashes = {}

            def snapshot(method):
                modified = pair.clone(); deleted = row['metrics'][method]['deleted_user_indices'][2]
                assert len(deleted) == len(row['metrics']['DT']['deleted_user_indices'][2])
                modified[1, [row['user_positions'][j] for j in deleted]] = int(ids[0, -1])
                digest = sha(modified[1].cpu().numpy().tobytes())
                assert digest == row['metrics'][method]['actual_input_hashes'][2]
                deleted_sets[method] = deleted; input_hashes[method] = digest
                values = {}; handles = []

                def retain(name, tensor):
                    assert name not in values
                    values[name] = tensor[1:2].detach().to('cpu', copy=True)

                for i, layer in enumerate(layers):
                    for name, module in [('input_norm', layer.input_layernorm), ('post_norm', layer.post_attention_layernorm)]:
                        def hook(_m, args, output, i=i, name=name):
                            retain(f'{i}/{name}_input', args[0]); retain(f'{i}/{name}_output', output)
                        handles.append(module.register_forward_hook(hook))
                    for name, module in [('mixer', layer.self_attn if layer.block_type == 'full_attention' else layer.linear_attn), ('mlp', layer.mlp), ('out', layer)]:
                        def hook(_m, _args, output, i=i, name=name):
                            retain(f'{i}/{name}', output[0] if isinstance(output, tuple) else output)
                        handles.append(module.register_forward_hook(hook))
                attention_indices = {id(layer.self_attn): i for i,layer in enumerate(layers) if layer.block_type=='full_attention'}
                active_gdn={}
                chunk=importlib.import_module('fla.ops.gated_delta_rule.chunk')
                stage_code=inspect.unwrap(chunk.chunk_gated_delta_rule_fwd).__code__
                fla_codes={inspect.unwrap(layer.linear_attn.chunk_gated_delta_rule).__code__ for layer in layers if layer.block_type=='linear_attention'}
                assert len(fla_codes)==1
                for i,layer in enumerate(layers):
                    if layer.block_type=='full_attention':
                        def gate(_m,_args,output,i=i,layer=layer):
                            h=layer.self_attn.config.num_attention_heads; d=layer.self_attn.head_dim
                            retain(f'{i}/gate',output.view(2,n,h,2*d)[...,d:])
                        def product(_m,args,i=i,layer=layer):
                            retain(f'{i}/product',args[0].view(2,n,layer.self_attn.config.num_attention_heads,layer.self_attn.head_dim))
                        handles.append(layer.self_attn.q_proj.register_forward_hook(gate))
                        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(product))
                        for coordinate in ('q','k'):
                            def norm_coordinates(_m,args,output,i=i,coordinate=coordinate):
                                retain(f'{i}/raw_'+coordinate,args[0]);retain(f'{i}/norm_'+coordinate,output)
                            handles.append(getattr(layer.self_attn,coordinate+'_norm').register_forward_hook(norm_coordinates))
                    else:
                        def norm_gate(_m,args,output,i=i,layer=layer):
                            h=layer.linear_attn.num_v_heads; d=layer.linear_attn.head_v_dim
                            assert len(args)==2 and args[0].numel()==args[1].numel()==output.numel()==2*n*h*d
                            retain(f'{i}/core',args[0].reshape(2,n,h,d)); retain(f'{i}/gate',args[1].reshape(2,n,h,d))
                            retain(f'{i}/product',output.reshape(2,n,h,d))
                        handles.append(layer.linear_attn.norm.register_forward_hook(norm_gate))
                        def gdn_enter(_m,_args,i=i):
                            assert not active_gdn;active_gdn['index']=i
                        def gdn_leave(_m,_args,_output,i=i):
                            assert active_gdn.pop('index')==i
                        handles.append(layer.linear_attn.register_forward_pre_hook(gdn_enter))
                        handles.append(layer.linear_attn.register_forward_hook(gdn_leave))
                def profile(frame,event,result):
                    if active_gdn:
                        i=active_gdn['index']
                        if frame.f_code in fla_codes and event=='call':
                            for coordinate in ('q','k'):retain(f'{i}/raw_'+coordinate,frame.f_locals[coordinate])
                            retain(f'{i}/core_g',frame.f_locals['g'])
                        if frame.f_code is stage_code and event=='return' and result is not None:
                            for coordinate in ('q','k'):retain(f'{i}/norm_'+coordinate,frame.f_locals[coordinate])
                            for coordinate in ('v','beta'):retain(f'{i}/core_'+coordinate,frame.f_locals[coordinate])
                    if frame.f_code is flash_attention_forward.__code__ and event=='call':
                        i=attention_indices[id(frame.f_locals['module'])]
                        for coordinate,name in [('q','query'),('k','key'),('v','value')]:retain(f'{i}/core_'+coordinate,frame.f_locals[name])
                    if frame.f_code is flash_attention_forward.__code__ and event=='return' and result is not None:
                        module=frame.f_locals['module']; assert id(module) in attention_indices
                        retain(f'{attention_indices[id(module)]}/core',result[0])
                def final(_m, args, output):
                    retain('final_norm_input', args[0]); retain('final_norm_output', output)
                handles.append(norm.register_forward_hook(final))
                try:
                    assert sys.getprofile() is None
                    sys.setprofile(profile)
                    with torch.no_grad():
                        out = model(input_ids=modified, attention_mask=mask, use_cache=False, logits_to_keep=selector.rows)
                        z = selector.pack_logits(out.logits)
                        lp = z.float().log_softmax(-1).gather(-1, selection.labels.repeat_interleave(2)[:, None]).squeeze(-1)
                        assert lp[0::2].cpu().tolist() == plain_detail['target_logp0']
                        partial_logprobs[method] = lp[1::2].cpu().tolist()
                        del out, z, lp
                finally:
                    sys.setprofile(None)
                    for handle in handles: handle.remove()
                assert len(values) == 32 * 17 + 2 and not active_gdn
                return values

            for method in ('DT', 'FT_K1'): partials[method] = timed(key + '/partial/' + method, lambda method=method: snapshot(method))

            class Observer:
                def __init__(self): self.boundaries = {}; self.layers = []
                def wants_decoder(self, index): return True
                def dot(self, coefficient, original, partial):
                    return float((coefficient.double() * (original[1::2].double() - partial.to('cuda').double())).sum())
                def boundary(self, label, m, values):
                    name = 'final_norm_output' if label == 'norm' else 'final_norm_input' if label == '32' else label + '/input_norm_input'
                    self.boundaries[label] = {method: self.dot(m, values, d[name]) for method, d in partials.items()}
                def decoder(self, index, d, c, e, mout, minput, terms):
                    rec = {'layer': index, 'mixer': layers[index].block_type, 'methods': {}}
                    module=layers[index].self_attn if layers[index].block_type=='full_attention' else layers[index].linear_attn
                    inner=terms['mixer']; mmid=terms['m_mixer_output']
                    if layers[index].block_type=='full_attention':
                        h=module.config.num_attention_heads; dim=module.head_dim
                        projection=_linear_transpose(mmid,module.o_proj.weight).reshape(selection.batch,n,h,dim)
                        core=c['attention_output']; gate=c['q_proj_output'].view(2,n,h,2*dim)[...,dim:]
                        product=c['o_proj_input'].reshape(2,n,h,dim)
                        mcore=inner['mcontent'].transpose(1,2); mraw=mcore; mgate=inner['mgate']
                    else:
                        h=module.num_v_heads; dim=module.head_v_dim
                        projection=inner['mnorm'].reshape(selection.batch,n,h,dim)
                        core=e['o']; gate=c['z']; product=c['norm_output'].reshape(2,n,h,dim)
                        mcore=inner['mo_native']; mraw=inner['mo_before_cast']; mgate=inner['mz']
                    fa_block=layers[index].block_type=='full_attention'
                    coefficients=inner['coeff']
                    if fa_block:
                        groups=module.num_key_value_groups;kh=h//groups
                        mqcore=coefficients['dq'].float()
                        mkcore=coefficients['dk'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                        mvcore=coefficients['dv'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                        cos,sin=terms['position_embeddings']
                        mqnorm=_partial_rotation_transpose(mqcore,cos[1::2],sin[1::2]).transpose(1,2)
                        mknorm=_partial_rotation_transpose(mkcore,cos[1::2],sin[1::2]).transpose(1,2)
                        mqraw=rmsnorm_secant_pullback(c['q_norm_input'][0::2].float(),c['q_norm_input'][1::2].float(),1+module.q_norm.weight.float(),mqnorm,module.q_norm.eps)
                        mkraw=rmsnorm_secant_pullback(c['k_norm_input'][0::2].float(),c['k_norm_input'][1::2].float(),1+module.k_norm.weight.float(),mknorm,module.k_norm.eps)
                        query_gate=torch.cat((mqraw,mgate),dim=-1).reshape(selection.batch,n,h*2*dim)
                        reconstructed=(_linear_transpose(query_gate,module.q_proj.weight)
                                      +_linear_transpose(mkraw.reshape(selection.batch,n,kh*dim),module.k_proj.weight)
                                      +_linear_transpose(mvcore.transpose(1,2).reshape(selection.batch,n,kh*dim),module.v_proj.weight))
                        boundary_difference=terms['m_mixer_input']-reconstructed
                        reconstruction={'relative_L2':float(boundary_difference.norm()/terms['m_mixer_input'].norm()),
                                        'max_absolute':float(boundary_difference.abs().max())}
                    else:
                        mqnorm=coefficients['q'];mknorm=coefficients['k']
                        mqraw=_l2_pullback(c['raw_q'][0::2],c['raw_q'][1::2],mqnorm)
                        mkraw=_l2_pullback(c['raw_k'][0::2],c['raw_k'][1::2],mknorm)
                        repeat=module.num_v_heads//module.num_k_heads
                        assert torch.equal(mqraw.reshape(selection.batch,n,module.num_k_heads,repeat,dim).sum(3),inner['mq'])
                        assert torch.equal(mkraw.reshape(selection.batch,n,module.num_k_heads,repeat,dim).sum(3),inner['mk'])
                        reconstruction={'grouped_coefficients_exact':True}
                    if fa_block:
                        lse=terms['native_lse']
                        ops={'q0':c['query'][1::2],'q1':c['query'][0::2],'k0':c['key'][1::2],'k1':c['key'][0::2],
                             'v0':c['value'][1::2],'u':inner['mcontent'],'lse0':lse[1::2],'lse1':lse[0::2]}
                        alternative=fa(ops,module.scaling,RightPaddedLengths([n],n,'cuda'))
                        alt_q=alternative['dq'].float()
                        alt_k=alternative['dk'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                        alt_v=alternative['dv'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                    else:
                        permutation=torch.arange(2*selection.batch,device='cuda')^1
                        assert all(value.shape[0]==2*selection.batch for value in e.values())
                        reversed_endpoints={name:value.index_select(0,permutation) for name,value in e.items()}
                        alternative=runner.finite_fla(reversed_endpoints,inner['mo_native'],terms['native_scale'])
                        alt_q,alt_k,alt_v=[alternative[k] for k in ('q','k','v')]
                        del reversed_endpoints
                    assert all(torch.isfinite(v).all() for v in alternative.values())
                    rec['normalization_coefficient_reconstruction']=reconstruction
                    for method, partial in partials.items():
                        def dot(m, actual, name): return self.dot(m, actual, partial[f'{index}/' + name])
                        mmid = terms['m_mixer_output']; ma = terms['m_mixer_input']; mn = terms['m_mlp_norm_output']
                        pin = dot(minput, d['input_norm_input'], 'input_norm_input')
                        residual1 = dot(mmid, d['input_norm_input'], 'input_norm_input')
                        norm1 = dot(ma, d['input_norm_output'], 'input_norm_output')
                        mix = dot(mmid, c['output'], 'mixer')
                        mid = dot(mmid, d['post_norm_input'], 'post_norm_input')
                        residual2 = dot(mout, d['post_norm_input'], 'post_norm_input')
                        norm2 = dot(mn, d['post_norm_output'], 'post_norm_output')
                        mlp = dot(mout, d['mlp_output'], 'mlp')
                        pout = dot(mout, d['output'], 'out')
                        errors = {'input_norm': pin - residual1 - norm1, 'mixer': norm1 - mix,
                                  'residual1_rounding': residual1 + mix - mid, 'post_norm': mid - residual2 - norm2,
                                  'MLP': norm2 - mlp, 'residual2_rounding': residual2 + mlp - pout}
                        assert abs(sum(errors.values()) - (pin - pout)) < 1e-7 * max(1., abs(pin), abs(pout))
                        pcore=dot(mcore,core,'core'); praw=dot(mraw,core,'core'); pgate=dot(mgate,gate,'gate')
                        pproduct=dot(projection,product,'product')
                        post={'output_projection':pproduct-mix,'output_gate_or_norm_gate':praw+pgate-pproduct,
                              'GDN_upstream_cast':pcore-praw,
                              'remaining_mixer':errors['mixer']-(pproduct-mix)-(praw+pgate-pproduct)-(pcore-praw)}
                        assert abs(sum(post.values())-errors['mixer'])<1e-7*max(1.,abs(errors['mixer']))
                        if fa_block:
                            pqraw=dot(mqraw,c['q_norm_input'],'raw_q');pkraw=dot(mkraw,c['k_norm_input'],'raw_k')
                            pqnorm=dot(mqnorm,c['q_norm_output'],'norm_q');pknorm=dot(mknorm,c['k_norm_output'],'norm_k')
                            pqcore=dot(mqcore,c['query'],'core_q');pkcore=dot(mkcore,c['key'],'core_k');pvcore=dot(mvcore,c['value'],'core_v')
                            pnative=dot(mcore.to(torch.bfloat16),core,'core')
                            internal={'Q_normalization':pqraw-pqnorm,'K_normalization':pkraw-pknorm,
                                      'rotary_rounding':pqnorm+pknorm-pqcore-pkcore,
                                      'finite_core':pqcore+pkcore+pvcore-pnative,'FA_upstream_cast':pnative-pcore,
                                      'compiled_boundary_reconstruction':dot(boundary_difference,c['input'],'input_norm_output')}
                        else:
                            pqraw=dot(mqraw,c['raw_q'],'raw_q');pkraw=dot(mkraw,c['raw_k'],'raw_k')
                            pqnorm=dot(mqnorm,e['q'],'norm_q');pknorm=dot(mknorm,e['k'],'norm_k')
                            pvcore=dot(coefficients['v'],e['v'],'core_v');pgcore=dot(coefficients['g'],e['raw_g'],'core_g');pbcore=dot(coefficients['beta'],e['beta'],'core_beta')
                            internal={'Q_normalization':pqraw-pqnorm,'K_normalization':pkraw-pknorm,
                                      'rotary_rounding':0.,'finite_core':pqnorm+pknorm+pvcore+pgcore+pbcore-pcore,
                                      'FA_upstream_cast':0.,'compiled_boundary_reconstruction':0.}
                        if fa_block:
                            prediction_alt=dot(alt_q,c['query'],'core_q')+dot(alt_k,c['key'],'core_k')+dot(alt_v,c['value'],'core_v')
                            actual_core=pnative
                        else:
                            prediction_alt=(dot(alt_q,e['q'],'norm_q')+dot(alt_k,e['k'],'norm_k')+dot(alt_v,e['v'],'core_v')
                                            +dot(alternative['g'],e['raw_g'],'core_g')+dot(alternative['beta'],e['beta'],'core_beta'))
                            actual_core=pcore
                        local_order={'actual_effect':actual_core,'current_prediction':internal['finite_core']+actual_core,
                                     'reversed_prediction':prediction_alt,'current_error':internal['finite_core'],
                                     'reversed_error':prediction_alt-actual_core}
                        internal['projection_conv_and_other']=post['remaining_mixer']-sum(internal.values())
                        assert abs(sum(internal.values())-post['remaining_mixer'])<1e-7*max(1.,abs(post['remaining_mixer']))
                        rec['methods'][method] = {'input_prediction':pin,'output_prediction':pout,'errors':errors,
                                                  'mixer_parts':post,'internal_parts':internal,'core_order_contrast':local_order,'mixer_boundary_predictions':{'core_raw':praw,'core_native':pcore,'gate':pgate,'product':pproduct,'output':mix}}
                    self.layers.append(rec)

            obs = Observer()
            observed, detail = timed(key + '/observed', lambda: trace_runner.attribute(pair, mask, selection, observer=obs))
            assert torch.equal(plain, observed)
            assert detail['target_logp0'] == plain_detail['target_logp0'] and detail['target_logp1'] == plain_detail['target_logp1']
            vectors[key + '/observed'] = observed.numpy()
            record = {'case': key, 'input_sha256': row['input_sha256'], 'deleted_sets': deleted_sets,
                      'partial_input_hashes': input_hashes, 'plain_observed_vector_equal': True,
                      'full_target_logprobs': plain_detail['target_logp1'], 'partial_target_logprobs': partial_logprobs,
                      'boundaries': obs.boundaries, 'layers': obs.layers, 'methods': {}}
            for method in ('DT', 'FT_K1'):
                actual = sum(float(x) - float(y) for x, y in zip(plain_detail['target_logp1'], partial_logprobs[method]))
                mass = float(plain[0, [row['user_positions'][j] for j in deleted_sets[method]]].sum())
                groups = {'head_seed': obs.boundaries['norm'][method] - actual,
                          'final_norm': obs.boundaries['32'][method] - obs.boundaries['norm'][method]}
                for layer in obs.layers:
                    replay_gap = layer['methods'][method]['output_prediction'] - obs.boundaries[str(layer['layer'] + 1)][method]
                    layer['methods'][method]['native_replay_boundary_gap'] = replay_gap
                    groups['native_replay_boundary'] = groups.get('native_replay_boundary', 0.) + replay_gap
                    for name, value in layer['methods'][method]['errors'].items():
                        family = layer['mixer'] if name == 'mixer' else name
                        groups[family] = groups.get(family, 0.) + value
                closing = sum(groups.values()) - (mass - actual)
                assert abs(obs.boundaries['0'][method] - mass) < 1e-7 * max(1., abs(mass))
                assert abs(closing) < 1e-6 * max(1., abs(mass), abs(actual)), closing
                record['methods'][method] = {'allocated_mass': mass, 'actual_drop': actual, 'prediction_error': mass - actual,
                                             'signed_family_errors': groups, 'telescope_closure': closing}
            report['cases'].append(record)
            for name, module in model.named_modules(): assert (type(module).forward, module.forward) == identity[name]
            np.savez_compressed(a.output / 'vectors.npz', **vectors); save()
            del partials, plain, observed, obs, detail, pair, ids, mask; gc.collect()
        report['status'] = 'complete'; report['vectors_sha256'] = sha((a.output / 'vectors.npz').read_bytes())
    except Exception:
        report['status'] = 'failed'; report['error'] = traceback.format_exc(); raise
    finally: save()


if __name__ == '__main__':
    main()
