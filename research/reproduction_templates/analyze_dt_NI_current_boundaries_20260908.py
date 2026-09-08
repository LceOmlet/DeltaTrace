"""Independent CPU audit of the latest NI current-control boundary capture."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_NI_current_boundaries_20260908_v1';started=time.perf_counter()
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();read=lambda x:json.loads(x.read_bytes());local=lambda x:A/'snapshot'/x.lstrip('/')
p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
assert r['status']=='NI_current_34boundaries_1DT4score_observation_complete'
assert p==r['protocol']==read(A/'dt_NI_current_boundaries_protocol_20260908.json')
for name,v in receipt['files'].items():assert sha(D/name)==v['sha256'] and (D/name).stat().st_size==v['bytes'],name
for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    for name in archive.namelist():assert archive.read(name)==(D/name).read_bytes(),name
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
assert sha(local(p['reference_results_path']))==p['reference_results_sha256']
assert sha(local(p['reference_vectors_path']))==p['reference_vectors_sha256']
prior=read(local(p['reference_results_path']));ref=np.load(local(p['reference_vectors_path']));z=np.load(D/'vectors.npz')
assert prior['status']=='all8FA_clean_secant_4DT84score_pilot_complete'
case=prior['cases']['niah_mq_q2_1'];frozen=case['curves']['control'];info=r['input'];assert info==case['input']
assert r['model_loads']==r['native_eager_diagnostics']==r['DT_calls_entered']==r['DT_calls']==1
assert r['scoring_forwards_entered']==r['scoring_forwards_returned']==4
assert r['FT_calls']==r['generation_calls']==r['extra_operator_calls']==0
assert r['finite_callback_counts']=={'finite_FA':{'entered':8,'returned':8},'finite_FLA':{'entered':24,'returned':24}}
assert r['DT_details']['norm_gate_rules']=={'0':'symmetric'} and not r['historical_boundary_coefficient_drift']['available']
kinds=[x['kind'] for x in r['DT_details']['calls']]
assert sum(k.startswith('native_replay_') for k in kinds)==32 and sum(k.startswith('finite_decoder_') for k in kinds)==32
assert sum(k.startswith('public_FA_LSE_') for k in kinds)==8
boundaries=[str(i) for i in range(33)]+['norm'];assert p['boundaries']==boundaries
assert set(r['boundary_coefficient_metadata'])==set(r['B2_boundary_contractions'])==set(boundaries)
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum()),'absolute':float(np.abs(x).sum())}
full=z['current_full'];evaluated=z['current_evaluated'];T,P=info['total_length'],info['prompt_length'];keep=set(info['keep'])
assert full.shape==(T,) and evaluated.shape==(P,) and np.array_equal(full[:P].astype(np.float32),evaluated)
assert np.isfinite(full).all()
delta=full-ref['niah_mq_q2_1_control_full'];close(np.linalg.norm(delta)/np.linalg.norm(ref['niah_mq_q2_1_control_full']),r['reference_vector_drift_report_only']['relative_L2'])
close(np.abs(delta).max(),r['reference_vector_drift_report_only']['max_absolute'])
edelta=evaluated.astype(np.float64)-ref['niah_mq_q2_1_control_evaluated'].astype(np.float64)
close(np.linalg.norm(edelta)/np.linalg.norm(ref['niah_mq_q2_1_control_evaluated']),r['evaluated_vector_drift_report_only']['relative_L2'])
hist=prior['runs'][3];assert hist['case']=='niah_mq_q2_1' and hist['method']=='control'
close(r['B2_actual_root_effect'],r['DT_details']['root_effect']);close(r['root_effect_minus_reference'],r['DT_details']['root_effect']-hist['details']['root_effect'])
close(r['seed_effect_minus_reference'],r['DT_details']['seed_effect']-hist['details']['seed_effect'])
out={'status':'independent_NI34boundary_audit_passed','analyzer_sha256':sha(Path(__file__)),'protocol_sha256':sha(D/'protocol.json'),
    'results_sha256':sha(D/'results.json'),'vectors_sha256':sha(D/'vectors.npz'),'job_seconds':r['job_seconds'],
    'counts':{'models':1,'eager_NI0':1,'DT':1,'decoder_replays':32,'finite_decoders':32,'auxiliary_FA':8,'finite_FA':8,'finite_FLA':24,'original_scorers':4,'FT':0,'extra_operators':0},
    'vector_drift':r['reference_vector_drift_report_only'],'evaluated_vector_drift':r['evaluated_vector_drift_report_only'],
    'root_effect_minus_reference':r['root_effect_minus_reference'],'seed_effect_minus_reference':r['seed_effect_minus_reference'],
    'historical_boundary_coefficient_drift':r['historical_boundary_coefficient_drift'],'points':{},
    'private_capture_scope':'34 actual coefficients and paired/current hidden inputs were captured; public NPZ contractions audited here. Private multi-GB tensors are not locally rehashed or replaced with historical tensors.'}
for step in [0,1,10,20]:
    row=r['points'][str(step)];assert row['input_receipt']==frozen['input_receipts'][step]
    close(row['prior_actual_native_score'],frozen['scores'][step]);close(row['native_score_minus_prior'],row['actual_native_score']-frozen['scores'][step])
    if step==0:
        out['points']['0']={k:row[k] for k in ['actual_native_score','prior_actual_native_score','native_score_minus_prior','B1_clean_vs_B2_input_boundary_relative_L2']};continue
    deleted=set(row['input_receipt']['deleted_positions']);assert deleted<=keep
    effect=r['points']['0']['actual_native_score']-row['actual_native_score'];pred=float(evaluated[sorted(deleted)].astype(np.float64).sum())
    close(effect,row['complete_input']['actual_effect']);close(pred,row['complete_input']['evaluated_prediction'])
    close(float(full[sorted(deleted)].sum()),row['complete_input']['full_signed_prediction'])
    groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
    arrays={name:z[str(step)+'_'+name] for name in row['coordinate_signed_groups']};assert all(x.shape==(1,T) and np.isfinite(x).all() for x in arrays.values())
    for name,x in arrays.items():
        saved=row['coordinate_signed_groups'][name]
        for k,v in stats(x).items():
            if k!='absolute':close(v,saved['total'][k])
        for group,idx in groups.items():
            assert saved['groups'][group]['count']==len(idx)
            for k,v in stats(x[0,idx]).items():
                if k!='absolute':close(v,saved['groups'][group][k])
    regions={}
    for left,right in zip(boundaries,boundaries[1:]):
        name=left+'_to_'+right;close(arrays[name],arrays['boundary_'+left]-arrays['boundary_'+right]);regions[name]=float(arrays[name].sum())
    for name in boundaries:close(arrays['boundary_'+name].sum(),row['coarse']['boundary_contractions'][name])
    regions['norm_to_actual_score']=float(arrays['boundary_norm'].sum())-effect;regions['evaluated_input_map']=pred-float(arrays['boundary_0'].sum())
    for name,value in regions.items():close(value,row['coarse']['regions'][name])
    close(sum(regions.values()),pred-effect);close(pred-effect,row['complete_input']['prediction_minus_actual'])
    positive=sorted(((k,v) for k,v in regions.items() if v>0),key=lambda kv:-kv[1]);negative=sorted(((k,v) for k,v in regions.items() if v<0),key=lambda kv:kv[1])
    out['points'][str(step)]={'complete_input':row['complete_input'],'actual_native_score':row['actual_native_score'],
        'native_score_minus_prior':row['native_score_minus_prior'],'regions':regions,'positive_terms_ranked':positive,'negative_terms_ranked':negative,
        'region_positive_sum':sum(v for v in regions.values() if v>0),'region_negative_sum':sum(v for v in regions.values() if v<0),
        'largest_region_coordinate_groups':{name:row['coordinate_signed_groups'][name] for name,_ in sorted(regions.items(),key=lambda kv:-abs(kv[1]))[:8] if name in row['coordinate_signed_groups']}}
    if step==20:out['points'][str(step)]['B1_allEOS_vs_B2_EOS_boundary_relative_L2']=row['B1_allEOS_vs_B2_EOS_boundary_relative_L2']
out['B2_regions']={left+'_to_'+right:r['B2_boundary_contractions'][left]-r['B2_boundary_contractions'][right] for left,right in zip(boundaries,boundaries[1:])}
out['interpretation']='Ranked boundaries localize signed conditional mismatch, not an identified internal operator defect. Head/seed includes scorer precision/conventions. Any next split must target the measured current boundary and preserve compensating negative terms; no speculative correction follows.'
out['audit_seconds']=time.perf_counter()-started;target=A/'dt_NI_current_boundaries_summary_20260908.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'output':str(target),'sha256':sha(target),'seconds':out['audit_seconds']}))
