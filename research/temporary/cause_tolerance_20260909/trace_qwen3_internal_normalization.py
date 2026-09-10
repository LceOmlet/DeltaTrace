"""Match Qwen3.5 fixed partial-deletion accounting with native Qwen3 DT."""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'environment', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--phase', choices=['pilot', 'remaining12'], required=True)
    a = p.parse_args()
    assert a.phase == 'pilot'
    reference = '/tmp/codex_clean_development16_20260909_v1/qwen3/results.json'
    digest = 'dccfbf8d2f2ba33b0ff68c686031fb86b2ac76501164f63d91228be545984de6'
    raw = Path(reference).read_bytes(); assert sha(raw) == digest
    rows = [r for r in json.loads(raw)['cases'] if (r['index'] < 2) == (a.phase == 'pilot')]
    rows.sort(key=lambda r: (r['dataset'], r['index']))
    assert len(rows) == (4 if a.phase == 'pilot' else 12)
    env = json.loads(a.environment.read_bytes())['qwen3']
    os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/deltatrace_clean_v1_inductor')
    for d in (env['official_root'], str(a.release/'deltatrace/clean/qwen3'), str(a.release/'deltatrace/accelerated')):
        sys.path.insert(0, d)
    import numpy as np
    import torch
    from exp.exp2 import run_exp as author
    from deferred import make_deferred_qwen3
    baseline, receipt = make_deferred_qwen3(a.release)
    from qwen3_internal_pair import propagate_paired_secant as instrument
    from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
    from vendor_fa_finite_runtime import VendorFAFiniteP1
    from flash_attn import flash_attn_func
    torch.set_num_threads(4); torch.manual_seed(73); torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(128, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit = max(512, torch._dynamo.config.accumulated_cache_size_limit)
    a.output.mkdir(exist_ok=False)
    derivation = json.loads((Path(__file__).parent/'qwen3_internal_trace_derivation.json').read_bytes())
    for name, info in derivation.items():
        assert sha((Path(__file__).parent/name).read_bytes()) == info['derived_sha256']
        assert sha((a.release/info['original_path']).read_bytes()) == info['original_sha256']
    report = {'status': 'loading', 'script_sha256': sha(Path(__file__).read_bytes()), 'phase': a.phase,
              'references': [[reference, digest]], 'baseline_sources': receipt, 'observer_sources': derivation,
              'cases': [], 'calls': [], 'generation_calls': 0, 'FT_calls': 0, 'metric_calls': 0}
    vectors = {}

    def save():
        q = a.output/'results.partial'; q.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); q.replace(a.output/'results.json')

    def timed(name, fn):
        report['status'] = name; save(); torch.cuda.synchronize(); start = time.perf_counter()
        value = fn(); torch.cuda.synchronize()
        report['calls'].append({'name': name, 'seconds': time.perf_counter()-start}); save(); return value

    try:
        model, tokenizer = timed('load', lambda: author.load_model(env['checkpoint'], 'cuda:0'))
        model.eval().requires_grad_(False); model.set_attn_implementation('flash_attention_2')
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        identities = {n: (type(m).forward, m.forward) for n,m in model.named_modules()}
        fa = VendorFAFiniteP1(env['finite_library'], env['finite_library_sha256'])
        layers = model.model.layers; assert len(layers) == 36
        for row in rows:
            key = f"{row['dataset']}_{row['index']}"
            ids = torch.tensor(row['input_ids'], device='cuda')[None]
            assert sha(ids.cpu().numpy().tobytes()) == row['input_sha256']
            assert int(ids[0,-1]) == tokenizer.eos_token_id
            base = ids.clone(); base[0, [row['user_positions'][j] for j in row['keep']]] = tokenizer.eos_token_id
            mask = torch.ones_like(ids)

            def attribute(observer=None):
                before, after = capture_checkpoint_pair(model, base, ids, mask, row['prompt_length'])
                fn = baseline if observer is None else instrument
                kw = {} if observer is None else {'observer': observer}
                result = fn(model, before, after, pv_rule='content_P1', finite_attention=fa, **kw)
                return result, before['target_logprobs32'].cpu().tolist(), after['target_logprobs32'].cpu().tolist()

            plain_detail, lp0, lp1 = timed(key+'/plain', attribute)
            plain = np.asarray(plain_detail['signed_full_sequence'], dtype=np.float64)[None]
            vectors[key+'/plain'] = plain
            partials = {}; partial_lp = {}; deleted_sets = {}; hashes = {}

            def snapshot(method):
                modified = torch.cat((base, ids), 0)
                deleted = row['metrics'][method]['deleted_user_indices'][2]
                assert len(deleted) == len(row['metrics']['DT']['deleted_user_indices'][2])
                modified[1, [row['user_positions'][j] for j in deleted]] = tokenizer.eos_token_id
                actual_hash = sha(modified[1].cpu().numpy().tobytes())
                assert actual_hash == row['metrics'][method]['actual_input_hashes'][2]
                deleted_sets[method] = deleted; hashes[method] = actual_hash
                values = {}; handles = []

                def retain(name, tensor):
                    assert name not in values
                    values[name] = tensor[1:2].detach().to('cpu', copy=True)

                for i, layer in enumerate(layers):
                    for name,module in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm)]:
                        def hook(_m, args, output, i=i, name=name):
                            retain(f'{i}/{name}_input', args[0]); retain(f'{i}/{name}_output', output)
                        handles.append(module.register_forward_hook(hook))
                    for name,module in [('mixer',layer.self_attn),('mlp',layer.mlp),('out',layer)]:
                        def hook(_m, args, output, i=i, name=name):
                            retain(f'{i}/{name}', output[0] if isinstance(output,tuple) else output)
                        handles.append(module.register_forward_hook(hook))
                active_attention={}
                for i,layer in enumerate(layers):
                    for coordinate in ('q','k'):
                        def coordinates(_m,args,output,i=i,coordinate=coordinate):
                            retain(f'{i}/raw_'+coordinate,args[0]);retain(f'{i}/norm_'+coordinate,output)
                        handles.append(getattr(layer.self_attn,coordinate+'_norm').register_forward_hook(coordinates))
                    def enter(_m,_args,i=i):
                        assert not active_attention;active_attention['index']=i
                    def leave(_m,_args,_output,i=i):
                        assert active_attention.pop('index')==i
                    handles.append(layer.self_attn.register_forward_pre_hook(enter))
                    handles.append(layer.self_attn.register_forward_hook(leave))
                def profile(frame,event,result):
                    if frame.f_code is flash_attn_func.__code__:
                        i=active_attention['index']
                        if event=='call':
                            for coordinate in ('q','k','v'):retain(f'{i}/core_'+coordinate,frame.f_locals[coordinate])
                        elif event=='return':
                            assert isinstance(result,torch.Tensor);retain(f'{i}/core_output',result)
                def final(_m,args,output):
                    retain('final_norm_input',args[0]); retain('final_norm_output',output)
                handles.append(model.model.norm.register_forward_hook(final))
                try:
                    assert sys.getprofile() is None
                    sys.setprofile(profile)
                    with torch.no_grad():
                        out = model(input_ids=modified, attention_mask=torch.ones_like(modified), use_cache=False)
                        z = out.logits[:,row['prompt_length']-1:-1]
                        labels = modified[:,row['prompt_length']:]
                        lp = z.float().log_softmax(-1).gather(-1,labels[:,:,None]).squeeze(-1)
                        assert lp[0].cpu().tolist() == lp0
                        partial_lp[method] = lp[1].cpu().tolist()
                        del out,z,lp
                finally:
                    sys.setprofile(None)
                    for h in handles: h.remove()
                assert len(values) == 36*15+2 and not active_attention
                return values

            for method in ('DT','FT_K1'):
                partials[method] = timed(key+'/partial/'+method, lambda method=method: snapshot(method))

            class Observer:
                def __init__(self): self.boundaries = {}; self.layers = []
                def dot(self,m,original,partial):
                    return float((m.double()*(original.double()-partial.to('cuda').double())).sum())
                def final(self,m,mfinal,after):
                    for name,coefficient,original,field in [('norm',mfinal,after['norm_out'],'final_norm_output'),('36',m,after['last'],'final_norm_input')]:
                        self.boundaries[name] = {method:self.dot(coefficient,original,d[field]) for method,d in partials.items()}
                def decoder(self,index,raw,mout,minput,mmid,ma_parts,mn_parts,inner):
                    rec = {'layer':index,'mixer':'full_attention','methods':{}}
                    groups=layers[index].self_attn.num_key_value_groups
                    b,t,kh,dim=raw['fa_k'].shape
                    dq=inner['dq'].transpose(1,2)
                    dk=inner['dk'].reshape(b,kh,groups,t,dim).sum(2).transpose(1,2)
                    dv=inner['dv'].reshape(b,kh,groups,t,dim).sum(2).transpose(1,2)
                    u=inner['u'].transpose(1,2)
                    for method,partial in partials.items():
                        def dot(m,original,name): return self.dot(m,original,partial[f'{index}/'+name])
                        pin = dot(minput,raw['x'],'input_norm_input')
                        residual1 = dot(mmid,raw['x'],'input_norm_input')
                        norm1 = sum(dot(m,raw['a'],'input_norm_output') for m in ma_parts)
                        mix = dot(mmid,raw['attn_out'],'mixer')
                        mid = dot(mmid,raw['mid'],'post_norm_input')
                        residual2 = dot(mout,raw['mid'],'post_norm_input')
                        norm2 = sum(dot(m,raw['mlp_in'],'post_norm_output') for m in mn_parts)
                        mlp = dot(mout,raw['mlp_out'],'mlp')
                        pout = dot(mout,raw['out'],'out')
                        errors = {'input_norm':pin-residual1-norm1,'mixer':norm1-mix,
                                  'residual1_rounding':residual1+mix-mid,'post_norm':mid-residual2-norm2,
                                  'MLP':norm2-mlp,'residual2_rounding':residual2+mlp-pout}
                        assert abs(sum(errors.values())-(pin-pout)) < 1e-7*max(1.,abs(pin),abs(pout))
                        pqraw=dot(inner['raw_q'],raw['qpre'],'raw_q');pkraw=dot(inner['raw_k'],raw['kpre'],'raw_k')
                        pqnorm=dot(inner['norm_q'],raw['qnorm'],'norm_q');pknorm=dot(inner['norm_k'],raw['knorm'],'norm_k')
                        pqcore=dot(dq,raw['fa_q'],'core_q');pkcore=dot(dk,raw['fa_k'],'core_k');pvcore=dot(dv,raw['fa_v'],'core_v')
                        poutput=dot(u,raw['fa_out'],'core_output');pnative=dot(u.to(torch.float16),raw['fa_out'],'core_output')
                        post={'output_projection':poutput-mix,'output_gate_or_norm_gate':0.,'GDN_upstream_cast':0.,
                              'remaining_mixer':errors['mixer']-(poutput-mix)}
                        internal={'Q_normalization':pqraw-pqnorm,'K_normalization':pkraw-pknorm,
                                  'rotary_rounding':pqnorm+pknorm-pqcore-pkcore,
                                  'finite_core':pqcore+pkcore+pvcore-pnative,'FA_upstream_cast':pnative-poutput,
                                  'compiled_boundary_reconstruction':0.}
                        internal['projection_conv_and_other']=post['remaining_mixer']-sum(internal.values())
                        assert abs(sum(post.values())-errors['mixer'])<1e-7*max(1.,abs(errors['mixer']))
                        assert abs(sum(internal.values())-post['remaining_mixer'])<1e-7*max(1.,abs(post['remaining_mixer']))
                        rec['methods'][method] = {'input_prediction':pin,'output_prediction':pout,'errors':errors,
                                                  'mixer_parts':post,'internal_parts':internal}
                    self.boundaries[str(index)] = {method:rec['methods'][method]['input_prediction'] for method in partials}
                    self.layers.append(rec)

            obs = Observer()
            detail, observed_lp0, observed_lp1 = timed(key+'/observed', lambda: attribute(obs))
            observed = np.asarray(detail['signed_full_sequence'],dtype=np.float64)[None]
            assert np.array_equal(plain,observed) and observed_lp0==lp0 and observed_lp1==lp1
            vectors[key+'/observed'] = observed
            record = {'case':key,'input_sha256':row['input_sha256'],'deleted_sets':deleted_sets,
                      'partial_input_hashes':hashes,'plain_observed_vector_equal':True,
                      'full_target_logprobs':lp1,'partial_target_logprobs':partial_lp,
                      'boundaries':obs.boundaries,'layers':obs.layers,'methods':{},
                      'plain_checks':plain_detail['deferred_validation'],'observed_checks':detail['deferred_validation']}
            for method in ('DT','FT_K1'):
                actual = sum(float(x)-float(y) for x,y in zip(lp1,partial_lp[method]))
                mass = float(plain[0,[row['user_positions'][j] for j in deleted_sets[method]]].sum())
                groups = {'head_seed':obs.boundaries['norm'][method]-actual,
                          'final_norm':obs.boundaries['36'][method]-obs.boundaries['norm'][method]}
                for layer in obs.layers:
                    gap = layer['methods'][method]['output_prediction']-obs.boundaries[str(layer['layer']+1)][method]
                    layer['methods'][method]['native_replay_boundary_gap'] = gap
                    groups['native_replay_boundary'] = groups.get('native_replay_boundary',0.)+gap
                    for name,value in layer['methods'][method]['errors'].items():
                        family = layer['mixer'] if name=='mixer' else name
                        groups[family] = groups.get(family,0.)+value
                closing = sum(groups.values())-(mass-actual)
                assert abs(obs.boundaries['0'][method]-mass) < 1e-7*max(1.,abs(mass))
                assert abs(closing) < 1e-6*max(1.,abs(mass),abs(actual)),closing
                record['methods'][method] = {'allocated_mass':mass,'actual_drop':actual,'prediction_error':mass-actual,
                                             'signed_family_errors':groups,'telescope_closure':closing}
            report['cases'].append(record)
            for name,module in model.named_modules(): assert (type(module).forward,module.forward)==identities[name]
            np.savez_compressed(a.output/'vectors.npz',**vectors); save()
            del partials,plain,observed,obs,detail,plain_detail,ids,base,mask; gc.collect()
        report['status'] = 'complete'; report['vectors_sha256'] = sha((a.output/'vectors.npz').read_bytes())
    except Exception:
        report['status'] = 'failed'; report['error'] = traceback.format_exc(); raise
    finally: save()


if __name__ == '__main__': main()
