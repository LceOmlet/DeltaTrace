"""Independent local receipt/NPZ algebra audit; no model, GPU, or new experiment."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_qk_softmax_diagnostic_20260908_v1'
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();read=lambda x:json.loads(x.read_bytes())
local=lambda remote:A/'snapshot'/remote.lstrip('/');started=time.perf_counter()
p=read(D/'protocol.json');r=read(D/'results.json');b=read(D/'build_results.json');receipt=read(D/'terminal_receipt.json')
assert r['status']==receipt['status']=='three_FA19_conditional_score_contractions_complete'
assert b['status']=='finite_extension_compiled_not_executed' and b['compile_returncode']==0
assert p==r['protocol']==b['protocol'] and not receipt['proc_exists'] and not receipt['build_proc_exists']
for name,v in receipt['files'].items():assert sha(D/name)==v['sha256'] and (D/name).stat().st_size==v['bytes'],name
for name,h in p['files_sha256'].items():assert sha(D/name)==h,name
for item in p['protected_sources']:assert sha(local(item['path']))==item['sha256'],item['path']
for archive in ['review_bundle.zip','build_review_bundle.zip']:
    with zipfile.ZipFile(D/archive) as z:
        for name in z.namelist():assert z.read(name)==(D/name).read_bytes(),name
assert sha(D/'build_results.json')==r['build_results_sha256']
assert b['library']['sha256']==r['diagnostic_library_sha256']
assert b['command'][-1].endswith('/'+p['diagnostic_library_name'])
assert b['command'][-3].endswith('/'+p['diagnostic_extension_name'])
assert ' T deltatrace_fa_finite_p1_bf16_d256_conditional_diag\n' in b['symbols']['stdout']
assert b['vendor_sources_before']==b['vendor_sources_after']
for name,h in b['vendor_sources_before'].items():assert sha(A.parent/'DeltaTrace/third_party/metax_fa_2_5_3'/name)==h,name
assert r['native_FA_entered']==r['native_FA_returned']==1
assert r['diagnostic_finite_entered']==r['diagnostic_finite_returned']==3 and r['diagnostic_phase_launches_expected']==9
for name in ['model_calls','DT_calls','scorer_calls','FT_calls','backward_calls']:assert r[name]==0
for name in ['native_model_forwards','native_model_backwards','attribution_calls','GPU_kernel_launches']:assert b[name]==0
assert r['public_B2_replay_drift']=={'bitwise_equal':True,'relative_L2':0.0,'max_absolute_difference':0.0}
h=read(local(p['hybrid_results_path']));assert h['status']=='eleven_native_FA_hybrid_contrasts_complete'
assert sha(D/'conditional_signed_rows.npz')==r['artifacts']['conditional_signed_rows.npz']['sha256']
z=np.load(D/'conditional_signed_rows.npz');hz=np.load(local(p['hybrid_vectors_path']))
terms=['coefficient_replay_difference','QK_interaction_and_multiplier_rounding','tile_weight_BF16_cast','finite_softmax_condition_and_native_precision']
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)))<1e-7
def stats(x):return {'net':float(x.sum()),'positive_sum':float(x.clip(min=0).sum()),'negative_sum':float(x.clip(max=0).sum()),'absolute_sum':float(np.abs(x).sum())}
out={'status':'independent_local_audit_passed','analyzer_sha256':sha(Path(__file__)),'source_results_sha256':sha(D/'results.json'),
    'source_protocol_sha256':sha(D/'protocol.json'),'build_results_sha256':sha(D/'build_results.json'),
    'diagnostic_library_sha256':r['diagnostic_library_sha256'],'build_seconds':b['wall_seconds'],'run_seconds':r['seconds'],
    'counts':{k:r[k] for k in ['native_FA_entered','native_FA_returned','diagnostic_finite_entered','diagnostic_finite_returned','model_calls','DT_calls','scorer_calls','FT_calls','backward_calls']},
    'binary_identity_scope':'Build-library SHA equals executed-library SHA in independently hashed build/run receipts. Diagnostic .so and private coefficient tensors were not downloaded; no claim of rehashing those local bytes.',
    'public_B2_replay_drift':r['public_B2_replay_drift'],'saved_B1_vs_B2_endpoint_drift':r['saved_B1_vs_B2_endpoint_drift'],'points':{}}
P,T=r['input']['prompt_length'],r['input']['total_length'];keep=set(r['input']['keep'])
for step in p['fixed_steps']:
    row=r['points'][step];hr=h['points'][step];v={name:z[step+'_'+name] for name in row['fields']}
    assert all(x.shape==(1,T) and np.isfinite(x).all() for x in v.values())
    assert row['input_receipt']==hr['input_receipt'] and all(row['coefficient_independent_of_actual_A'].values())
    for drift in row['coefficient_replay_drift'].values():assert drift=={'bitwise_equal':True,'relative_L2':0.0,'max_absolute_difference':0.0}
    close(v['saved_qk_prediction'],hz[step+'_qk_prediction']);close(v['R0'],hz[step+'_R0'])
    for name,x,y in zip(terms,['saved_qk_prediction','replayed_qk_prediction','T_BF16','T_FP32'],['replayed_qk_prediction','T_BF16','T_FP32','R0']):close(v[name],v[x]-v[y])
    close(sum(v[k] for k in terms),v['saved_qk_prediction']-v['R0'])
    route=float((v['saved_qk_prediction']-v['R0']).sum());close(route,row['original_route_prediction_error']);close(route,p['original_route_error'][step])
    close(route,hr['fields']['routing_at_baseline_values_prediction_error']['net'])
    deleted=set(row['input_receipt']['deleted_positions']);assert deleted<=keep
    groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
    assert sorted(i for ids in groups.values() for i in ids)==list(range(T))
    for name,x in v.items():
        for key,value in stats(x).items():close(value,row['fields'][name][key])
        for group,ids in groups.items():
            assert row['groups'][group]['count']==len(ids)
            for key,value in stats(x[0,ids]).items():close(value,row['groups'][group]['fields'][name][key])
    seed=hr['BF16_seed_contrast_differences']['R0']
    close(hz[step+'_core_error_at_stored_seed']-hz[step+'_core_error_at_BF16_seed'],hz[step+'_BF16_minus_stored_seed_actual_effect'])
    out['points'][step]={'original_route_error':route,'four_terms':{k:row['fields'][k] for k in terms},
        'four_term_groups':{group:{k:row['groups'][group]['fields'][k] for k in terms} for group in groups},
        'actual_contractions':{k:row['fields'][k] for k in ['saved_qk_prediction','T_BF16','T_FP32','R0']},
        'coefficient_replay_drift':row['coefficient_replay_drift'],'BF16_minus_FP32_seed_R0_effect_from_hybrid':seed,
        'softmax_term_if_R0_uses_BF16_seed':row['fields'][terms[-1]]['net']-seed['net']}
out.update(precision_scope=r['precision_scope'],group_contract=r['group_contract'],allEOS_scope=r['allEOS_scope'],
    audit_seconds=time.perf_counter()-started,total_build_run_seconds=b['wall_seconds']+r['seconds'],
    conclusion='Mid route mismatch is dominated by finite-softmax-condition/native-precision term, not tile cast or coefficient replay. It still includes B1/B2 transfer and native precision; no candidate or new metric result.')
target=A/'dt_fa19_qk_softmax_diagnostic_summary_20260908.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'output':str(target),'sha256':sha(target),'seconds':out['audit_seconds'],'mid_terms':out['points']['10']['four_terms']}))
