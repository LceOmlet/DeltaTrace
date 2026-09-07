"""Verify source, real FA dispatch, complete vectors, batch layout and matched cost."""
import hashlib,json,statistics,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_fa_native_strides_integration_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
with zipfile.ZipFile(F/'review_bundle.zip') as z:
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (F/name).write_bytes(z.read(name))
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert p==json.loads((F/'protocol.json').read_text())
assert sha(F/'study.py')==p['study_sha256']
for name,digest in p['sources'].items():assert sha(F/name)==digest
assert d['status']=='FA_native_strides_B1_and_real_B4_exact'
assert d['checkpoint_before']==d['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']
assert d['fresh_attributions']==d['native_root_forwards']==12
assert d['native_vjps']==d['evaluation_forwards']==d['ft_attribution_forwards']==d['quality_queries']==0
assert d['extra_layer_replay_calls']==d['extra_native_fa_attention_calls']==d['finite_FA_calls_enqueued']==432
assert d['native_attribution_endpoint_trajectories']==60
out={'status':'FA_native_strides_B1_and_real_B4_verified','raw_sha256':sha(F/'results.json'),
    'protocol_sha256':sha(F/'protocol.json'),'library_sha256':p['strided_library_sha256'],
    'attributions':12,'quality_queries':0,'VJPs':0,'FT_attributions':0,'generated_targets':0,
    'native_root_forwards':12,'native_endpoint_trajectories':60,'decoder_replays':432,
    'auxiliary_public_FA_calls':432,'finite_attention_calls':432,
    'groups':[],'job_seconds':d['job_seconds'],'goal_complete':False}
numeric=[]
for index,g in enumerate(d['integration_groups']):
    assert g['selection']==p['integration_groups'][index]
    batch=g['example_batch_size'];assert batch in [1,4] and len(g['runs'])==6
    for repeat in range(3):
        old,new=[next(r['result'] for r in g['runs'] if r['mode']==mode and r['repeat']==repeat) for mode in ['old','native_strides']]
        assert np.array_equal(old['signed_full_sequence'],new['signed_full_sequence'])
        assert old['endpoint_scores32']==new['endpoint_scores32']
        for mode,result in [('old',old),('native_strides',new)]:
            cost=result['end_to_end_cost']
            assert cost['native_forwards']==1 and cost['native_forward_trajectories']==batch*2
            assert cost['extra_replay_calls']==36
            assert cost['public_FA_activity']['auxiliary_attempts']==cost['public_FA_activity']['auxiliary_completed']==36
            for item in cost['public_FA_activity']['metadata']:
                assert item['testing_return']['numel'] in [0,None]
            boundaries=result['native_layer_boundary_checks']
            if isinstance(boundaries,dict):boundaries=boundaries['paired_batch']
            for item in boundaries:assert item['native_input_exact'] and item['native_output_exact']
            finite=cost['finite_FA_activity'];assert len(finite)==36
            for item in finite:
                assert item['calls_attempted']==item['calls_enqueued']==1
                shapes=[v['shape'] for v in item['buffer_contract']]
                assert shapes[0][0]==batch and shapes[0][1]==32 and len(shapes)==13
                for pos in [1,3,4]:assert shapes[pos][1]==8
                assert shapes[10]==shapes[11]==shapes[12]==shapes[0]
                assert item['global_endpoint_mean_buffers']==0 and item['extra_shared_tile'] is False
                assert item['GQA_input_expansion'] is False and item['kv_heads']==8
                assert item['endpoint_mean']=='FP32_add_FP16_store_in_FA_shared_tile_from_existing_loads'
                if mode=='native_strides':
                    assert item['QKV_copy_bytes']==0
                    for name in ['q0','k0','q1','k1','v0']:
                        assert item['input_storage_reused'][name]
                        assert item['input_strides'][name]==item['kernel_input_strides'][name]
                    assert item['input_storage_reused']['u'] is False
                    for v in item['buffer_contract'][10:]:
                        b,h,n,dim=v['shape'];assert v['strides']==[h*n*dim,n*dim,dim,1]
        assert g['comparisons'][repeat]['full_signed_exact'] and g['comparisons'][repeat]['endpoints_exact']
    measured={m:[r['result']['seconds'] for r in g['runs'] if r['mode']==m and not r['warm']] for m in ['old','native_strides']}
    peaks={m:[r['result']['peak_allocated_bytes'] for r in g['runs'] if r['mode']==m and not r['warm']] for m in measured}
    assert all(len(v)==2 for v in measured.values())
    out['groups'].append({'selection':g['selection'],'example_batch_size':batch,
        'all_same_job_full_vectors_exact':True,'endpoints_exact':True,'sign_flips':0,
        'measured_seconds':measured,'peak_bytes':peaks,
        'new_to_shared_mean_median_ratio':statistics.median(measured['native_strides'])/statistics.median(measured['old']),
        'timing_scope':'Two measured pairs per shape; same-job shared-mean-reuse baseline. All warm and measured calls retained in numeric evidence. No FT comparison.'})
    numeric.append(g)
profile=d['native_strides_B4_profile'];assert sha(F/profile['file'])==profile['sha256']
events=json.loads((F/profile['file']).read_text())['traceEvents']
kernels=Counter(e['name'] for e in events if e.get('cat')=='kernel')
finite={k:n for k,n in kernels.items() if 'deltatrace_fa_finite_p1_kernel_strided' in k}
native={k:n for k,n in kernels.items() if 'flash' in k.lower() and 'fwd' in k.lower() and k not in finite}
assert sum(finite.values())==sum(native.values())==108
out['actual_B4_profile']={'sha256':profile['sha256'],'finite_kernels':finite,'native_FA_kernels':native,
    'no_new_dense_attention_source_path':True}
scopes=[e for e in events if e.get('cat')=='user_annotation' and e['name']=='ATTR_VENDOR_FA_FINITE_P1']
assert len(scopes)==36
cpu={e.get('args',{}).get('External id'):e for e in events if e.get('cat') in ['cpu_op','user_annotation'] and 'External id' in e.get('args',{})}
wrapper=[]
for e in events:
    if e.get('cat')!='kernel' or 'deltatrace_fa_finite_p1_kernel' in e['name']:continue
    c=cpu.get(e.get('args',{}).get('External id'))
    if c is not None and any(x['ts']<=c['ts'] and c['ts']+c.get('dur',0)<=x['ts']+x['dur']+1e-3 for x in scopes):wrapper.append(e)
out['FA_wrapper_profile']={'GPU_kernel_count':len(wrapper),'GPU_seconds':sum(x['dur'] for x in wrapper)/1e6,
    'kernel_names':dict(Counter(x['name'] for x in wrapper)),
    'scope':'Existing-budget profiled B4 warm. Sum of kernel durations is not end-to-end latency.'}
assert len(wrapper)==36, out['FA_wrapper_profile']
out['groups'][0]['timing_caveat']='Shared-mean baseline first measured1.745s versus second0.951s is anomalous; no28.6percent speedup claim. All timings preserved. Similar first-measured NI baseline spikes occurred in previous jobs; the cause is unresolved, so this is not treated as evidence for speed.'
out['promotion_decision']='Retain as a verified address/layout candidate, not the preferred performance path. Both measured B4 pairs were slower and full peak was unchanged;2 measured pairs are too few for a broad regression claim. Current preferred FA path remains shared-mean reuse.'
out['remaining_FA_work']=['Three finite score/weight passes per layer remain; reducing them must preserve the signed finite expression and acceptable coefficient quantization.',
    'Each decoder replay still has one extra unchanged public FA call for documented LSE. A supported way to reuse metadata must be established before removing it.',
    'One U conversion remains before the finite operator; three output FP32 conversions remain before already-compiled GQA/RoPE. Reuse existing public compiler boundaries if worthwhile; no separate attention implementation.']
out['conclusion']='Native FA stride addressing is integrated and verified for original B1/B4 full signed vectors/endpoints. QKV reuse actual storage; preparation falls from7 to1 kernels/layer. Three finite passes and actual default model FA remain. Matched times/peaks are reported without FT, broad-performance, long-input or independent-quality claims.'
(A/'fa_native_strides_integration_summary_20260907.json').write_text(json.dumps(out,indent=2))
(A/'fa_native_strides_integration_numeric_20260907.json').write_text(json.dumps({'groups':numeric},separators=(',',':')))
print(json.dumps(out))
