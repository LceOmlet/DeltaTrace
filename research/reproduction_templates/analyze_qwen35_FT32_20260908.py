"""Audit saved complete FT hop vectors and original CPU recovery, no model calls."""
import ast,hashlib,json,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
def load(name):
    d=A/('snapshot${ARTIFACT_ROOT}/codex_'+name+'_20260908_v1')
    terminal=json.loads((d/'terminal_receipt.json').read_bytes());assert terminal['handle_missing']
    assert sha((d/'review_bundle.zip').read_bytes())==terminal['bundle_sha256']
    with zipfile.ZipFile(d/'review_bundle.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            f=(d/name).resolve();assert f.is_relative_to(d.resolve());f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(name))
    raw=(d/'results.json').read_bytes();assert sha(raw)==terminal['results_sha256'];r=json.loads(raw)
    for name,digest in r['protocol']['files_sha256'].items():assert sha((d/name).read_bytes())==digest;ast.parse((d/name).read_bytes())
    for a in r.get('artifacts',[]):
        if a['download']:assert sha((d/a['file']).read_bytes())==a['sha256']
    return d,r,sha(raw)
old_d,old,old_sha=load('qwen35_FT32');d,r,rsha=load('qwen35_FT32_resume');nd,n,nsha=load('qwen35_saved_recovery')
assert old['status']=='failed' and old['content_calls']==34 and r['status']=='corrected_FT32_hops0_to3_resumed_with_author_source_bounds'
assert r['content_calls']==97 and r['decoder_replays']==0 and old['decoder_replays']==30
assert old['sources_before']==r['sources_before']==r['sources_after']
assert n['status']=='five_fixed_methods_original_NI0_recovery_executed_CPU_only' and n['recovery_calls']==5 and n['GPU_calls']==0
layers_old=np.load(old_d/'hop0_FT_scores.npz')['layer_token_scores'];hops=[]
for h in range(4):
    a=dict(np.load(d/f'hop{h}_FT_scores.npz'));assert all(np.isfinite(v).all() for v in a.values())
    total=np.zeros((2,605),dtype=np.float32)
    for i in range(32):total+=a['layer_token_scores'][i]
    assert np.array_equal(total,a['token_total'])
    limits=[604,367] if h==0 else [566,346]
    assert all(np.count_nonzero(a['layer_token_scores'][:,b,end:])==0 for b,end in enumerate(limits))
    if h==0:assert np.array_equal(a['layer_token_scores'][:31],layers_old[:31])
    assert np.count_nonzero(a['observation_sum'][0,357:])==np.count_nonzero(a['observation_sum'][1,227:])==0
    hops.append(a)
nodes=Counter()
for result in [old,r]:
    for hop in result['hops'].values():
        for row in hop['layers'].values():nodes.update(x['node'] for x in row.get('native_node_events',[]))
assert nodes=={'FlashAttnVarlenFuncBackward':33,'ChunkGatedDeltaRuleFunctionBackward':98,'CausalConv1dFnBackward':98}
range_review={}
for h,i in [(0,31),(1,1)]:
    a=dict(np.load(d/f'hop{h}_layer{i}_source_range_review.npz'))
    assert all(np.isfinite(x).all() for x in a.values())
    assert all(np.count_nonzero(a['bounded_scores'][b,int(end):])==0 for b,end in enumerate(a['limits']))
    range_review[f'hop{h}_layer{i}']=r['hops'][str(h)]['layers'][str(i)]
vectors=dict(np.load(nd/'immutable_score_vectors.npz'));assert sha((nd/'immutable_score_vectors.npz').read_bytes())==n['vector_sha256']
spans=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json').read_bytes())['cases'][0]
user=spans['input_metadata']['author_user_positions'];keep=spans['mapping']['keep_local_indices'];gold=set(spans['mapping']['gold_user_token_indices'])&set(keep)
for method,row in n['methods'].items():
    selected=row['selected_user_indices'];assert len(selected)==31 and len(set(selected))==31 and set(selected)<=set(keep)
    assert row['recovery']==len(set(selected)&gold)/len(gold)
    used=np.maximum(vectors[method][user],0);unselected=sorted(set(keep)-set(selected))
    assert used[selected].min()>=used[unselected].max()
    if method.startswith('FT_'):
        h=int(method[-1]);assert np.array_equal(vectors[method],hops[h]['observation_sum'][0])
costs={}
for phase,result in [('initial_failed',old),('selective_resume',r)]:
    groups={}
    for c in result['calls']:
        key=('native_FA' if 'public_FA_' in c['kind'] else 'native_FLA' if 'public_FLA_' in c['kind'] else
            'native_conv' if 'public_linear_conv_' in c['kind'] else 'FT_projection' if 'FT_projection_' in c['kind'] else
            'original_decoder_replay' if 'replay_with_capture' in c['kind'] else 'weights_cache_IO')
        v=groups.setdefault(key,{'calls':0,'seconds':0.,'max_stage_bytes':0});v['calls']+=1;v['seconds']+=c['seconds'];v['max_stage_bytes']=max(v['max_stage_bytes'],c['peak_bytes'])
    costs[phase]={'job_seconds_before_bundle':result['job_seconds_before_bundle'],'timed_seconds':sum(x['seconds'] for x in result['calls']),'groups':groups}
s={'status':'same_model_FT0_3_complete_first_NI0_quality_negative_for_DeltaTrace',
    'raw_sha256':rsha,'initial_failure_sha256':old_sha,'recovery_raw_sha256':nsha,
    'source_sha256':r['protocol']['files_sha256'],'protocol_sha256':sha((d/'protocol.json').read_bytes()),
    'native_nodes_all_attempts':dict(nodes),'total_content_calls_all_attempts':131,
    'original_decoder_replays':30,'saved_decoder_reuses':2,'reused_FT0_layer_scores':31,
    'author_hop_controller_reference':r['author_hop_controller_reference'],'thinking_ratios':{h:v['ratio_after'] for h,v in r['hops'].items()},
    'range_correction':range_review,'costs':costs,'source_cache_bytes':sum(x['bytes'] for x in old['artifacts'] if x['file'].endswith('.pt')),
    'recovery':n,'all_remote_jobs_terminal':True,'RISE_MAS_executed':False,
    'decision':'Engineering budget closed. Do not pursue tiny rounding/bitwise/framework polish. First NI0 recovery is DeltaTrace2.5%,FT0 5%,FT1-3 7.5%; no quality advantage. Next bounded original RISE/MAS on same NI0/MH1 and frozen five methods. No method retuning or dataset expansion from this one result.',
    'limits':['One historically used NI example; not independent confirmation or paper-table reproduction.',
        'Explicit corrected native-content FT variant on sameQwen3.5-9B; unchanged author variant identity preserved separately.',
        'Original recovery positive-clamps the signed DeltaTrace vector; this metric does not test deletion signs.',
        'Costs include failed/repeated work, cache IO and cold native calls; not matched warm whole-method FT/DeltaTrace timing.']}
(A/'qwen35_FT32_summary_20260908.json').write_text(json.dumps(s,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':s['status'],'recovery':{k:v['recovery'] for k,v in n['methods'].items()},'native_nodes':dict(nodes),'costs':costs}))
