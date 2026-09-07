"""Audit full GDN vectors and the real native gradient limit; no GPU calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
folders=['qwen35_gdn_integration_20260908_v1','qwen35_gdn_integration_20260908_v2',
    'qwen35_gdn_saved_recovery_20260908_v1','qwen35_gdn_dtype_matched_20260908_v1']
dirs=[A/'snapshot/tmp'/('codex_'+name) for name in folders];results=[]
for directory in dirs:
    with zipfile.ZipFile(directory/'review_bundle.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            path=(directory/name).resolve();assert path.is_relative_to(directory.resolve())
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
    r=json.loads((directory/'results.json').read_bytes());p=json.loads((directory/'protocol.json').read_bytes());assert r['protocol']==p
    for name,digest in p['files_sha256'].items():assert sha((directory/name).read_bytes())==digest
    for item in r['artifacts']:
        if item.get('location')!='remote_only':assert sha((directory/item['file']).read_bytes())==item['sha256']
    results.append(r)
failed,partial,unmatched,recovery=results;D=dirs[-1];p=recovery['protocol']
assert failed['status']==partial['status']=='failed'
assert failed['root_forward_attempts']==1 and failed['root_forwards_completed']==failed['finite_attempts']==0
assert partial['root_forwards_completed']==1 and partial['finite_completed']==2 and partial['native_GDN_backward_attempts']==1
assert recovery['status']=='dtype_matched_GDN_vectors_and_native_gradient_limit_recovered'
assert sha((dirs[2]/'results.json').read_bytes())==p['unmatched_recovery_sha256']
assert recovery['official_loading_policy']['dtype_plan']=={}
assert set(recovery['official_loading_policy']['empty_parameter_dtypes'].values())=={'torch.bfloat16'}
assert all(v['dtype']=='torch.bfloat16' for v in recovery['weight_tensor_receipts'].values())
assert sum(v['dtype']=='torch.float32' for v in unmatched['weight_tensor_receipts'].values())==2
assert recovery['finite_attempts']==2 and recovery['native_backward_attempts']==recovery['native_forward_attempts']==1
assert all(recovery[k]==0 for k in ['full_model_loads','full_model_forwards','generation_calls','quality_queries'])
assert all(x['generation_calls']==x['quality_queries']==0 for x in results)
assert recovery['sources_before']==recovery['sources_after'] and recovery['weight_stats_before']==recovery['weight_stats_after']
assert sha((dirs[1]/'results.json').read_bytes())==p['parent_sha256']
assert sha((dirs[0]/'results.json').read_bytes())==partial['protocol']['prior_failed_result_sha256']
event,=recovery['native_conv_backward_node_events']
assert event['returned_input_gradient'] and event['thread_id']!=event['main_thread']
def load(name):
    with np.load(D/(name+'.npz'),allow_pickle=False) as z:return {k:z[k].astype(np.float64) for k in z.files}
def metrics(a,b):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    den=float(np.linalg.norm(a.ravel()));err=float(np.linalg.norm((b-a).ravel()))
    return {'relative_L2':err/den if den else None,'max_abs':float(np.abs(b-a).max()),
        'reference_norm':den,'cosine':float(np.dot(a.ravel(),b.ravel())/(den*np.linalg.norm(b.ravel()))) if den else None}
actual=load('actual');native=load('native');equal=load('equal')
mask=actual['mask'].astype(bool);assert mask.sum(1).tolist()==[605,368]
seed=actual['output1']*mask[:,:,None]
observed=seed*(actual['output1']-actual['output0'])
contributions=actual['finite']*(actual['input1']-actual['input0'])
reference=observed.sum((1,2));allocated=contributions.sum((1,2))
total=metrics(reference,allocated)
total.update(reference=reference.tolist(),allocated=allocated.tolist(),residual=(allocated-reference).tolist(),
    sample_relative_residual=((allocated-reference)/reference).tolist())
parent_allocated=np.array(partial['boundary_effects']['GDN_total']['allocated'])
eqmetrics=metrics(native['gradient'],equal['finite'])
root_outputs=np.stack([actual['output0'],actual['output1']],axis=1).reshape(4,*actual['output0'].shape[1:])
padding={'actual_max_abs':float(np.abs(actual['finite'][~mask]).max()),
    'equal_max_abs':float(np.abs(equal['finite'][~mask]).max()),'native_max_abs':float(np.abs(native['gradient'][~mask]).max())}
token_contributions=contributions.sum(-1)
s={'status':'complete_local_GDN_finite_vectors_and_native_limit_audited_whole_model_pending',
    'raw_sha256':{name:sha((directory/'results.json').read_bytes()) for name,directory in zip(folders,dirs)},
    'protocol_sha256':{name:sha((directory/'protocol.json').read_bytes()) for name,directory in zip(folders,dirs)},
    'runtime_sha256':p['files_sha256']['qwen35_gdn_finite.py'],'native_sources_unchanged':True,
    'actual_input_shape':list(actual['finite'].shape),'valid_lengths':mask.sum(1).tolist(),
    'total_finite_effect':total,'recovered_vs_first_run_total_allocation':metrics(parent_allocated,allocated),
    'equal_endpoints_vs_native':eqmetrics,
    'equal_endpoints_vs_native_per_sample':[metrics(native['gradient'][b],equal['finite'][b]) for b in range(2)],
    'native_replay_output_vs_root':metrics(root_outputs,native['replay_output']),
    'all_native_replay_boundaries_vs_root':recovery['native_replay_vs_real_root'],
    'original_loading_policy':recovery['official_loading_policy'],
    'excluded_wrong_dtype_recovery':{'raw_sha256':sha((dirs[2]/'results.json').read_bytes()),
        'reason':'FP32 checkpoint norm.weight/A_log were not converted according to the original BF16 model loader; not a target-model result.',
        'separate_summary':'qwen35_gdn_unmatched_recovery_summary_20260908.json'},
    'padding':padding,'boundary_effects':partial['boundary_effects'],
    'valid_token_sign_counts':[{'positive':int((token_contributions[b,mask[b]]>0).sum()),
        'negative':int((token_contributions[b,mask[b]]<0).sum()),'zero':int((token_contributions[b,mask[b]]==0).sum())} for b in range(2)],
    'native_backward_observation':{'main_thread_missed':True,'node_hook_confirmed':True,'callback_on_different_thread':True},
    'local_finite_calls':partial['finite_calls'],'recovery_calls':recovery['calls'],
    'root_diagnostic_seconds':partial['root_diagnostic_seconds'],'root_diagnostic_peak_bytes':partial['root_diagnostic_peak_bytes'],
    'compiler_counters':recovery['compiler_counters'],
    'budget':{'full_model_loads':2,'root_forward_attempts':2,'completed_root_forwards':1,
        'standalone_original_GDN_module_loads':2,'full_GDN_finite_calls':6,'local_native_GDN_forward_backward_calls':3,
        'metadata_only_model_constructions':1,
        'auxiliary_paired_linear_conv_forwards':6,'auxiliary_paired_linear_conv_backwards':6,
        'quality_queries':0,'generation_calls':0,'whole_model_attributions':0,
        'extra_calls_beyond_initial_screen':{'root_forward_attempt':1,'finite_GDN':3,'native_GDN_forward_backward':2},
        'inconclusive_CPU_only_import_probe':1},
    'limitations':['Only layer0 and one fixed local energy cotangent on two historical official cases; not the model answer target.',
        'Aggregate residuals can cancel. Per-head FLA residuals and every boundary remain visible; no new precision threshold.',
        'Ordinary native timings include local forward and passive CPU capture. They are not comparable to finite-only warm time.',
        'Root checkpoint and endpoint captures remain remote with recorded hashes. Two observer failures and one wrong-dtype recovery are preserved; the latter is excluded from target-model claims.',
        'Compiler tuning totals are incomplete because v2 ended before exporting counters. Recovery cache counters cannot stand in for the entire family.',
        'Full attention D256/BF16, remaining decoder/MLP/norm propagation, original metrics and complete attribution costs are not done.']}
(A/'qwen35_gdn_integration_summary_20260908.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps({k:s[k] for k in ['actual_input_shape','total_finite_effect','equal_endpoints_vs_native','equal_endpoints_vs_native_per_sample',
    'native_replay_output_vs_root','recovered_vs_first_run_total_allocation','padding','local_finite_calls','compiler_counters']},ensure_ascii=False))
