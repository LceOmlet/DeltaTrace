"""Independent CPU receipt and algebra audit; no candidate/model execution."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_supported_secant_20260908_v1';started=time.perf_counter()
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();read=lambda x:json.loads(x.read_bytes());local=lambda x:A/'snapshot'/x.lstrip('/')
p=read(D/'protocol.json');r=read(D/'results.json');b=read(D/'build_results.json');receipt=read(D/'terminal_receipt.json')
assert p==read(A/'dt_fa19_supported_secant_protocol_20260908.json')==r['protocol']==b['protocol']
assert r['status']=='one_supported_secant_saved_layer_candidate_observed'
assert not receipt['proc_exists'] and not receipt['build_proc_exists']
assert b['status']=='finite_extension_compiled_not_executed' and b['compile_returncode']==0
for name,v in receipt['files'].items():assert sha(D/name)==v['sha256'] and (D/name).stat().st_size==v['bytes'],name
for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
for item in p['protected_sources']:
    path=local(item['path'])
    if path.is_file():assert sha(path)==item['sha256'],item['path']
for archive in ['review_bundle.zip','build_review_bundle.zip']:
    with zipfile.ZipFile(D/archive) as z:
        for name in z.namelist():assert z.read(name)==(D/name).read_bytes(),name
assert sha(D/'build_results.json')==r['build_results_sha256'] and b['vendor_sources_before']==b['vendor_sources_after']
assert b['command'][-1].endswith('/'+p['diagnostic_library_name'])
assert ' T deltatrace_fa_finite_p1_supported_secant\n' in b['symbols']['stdout']
assert r['candidate_finite_entered']==r['candidate_finite_returned']==1 and r['dv_bitwise_equal_to_original']
for name in ['native_FA_calls','model_calls','DT_calls','scorer_calls','FT_calls','backward_calls']:assert r[name]==0
assert all(v['finite']==v['total'] for v in r['finite_element_counts'].values())
assert sha(D/'candidate_signed_rows.npz')==r['signed_rows']['sha256']
z=np.load(D/'candidate_signed_rows.npz');hz=np.load(local(p['hybrid_vectors_path']));source=read(local(p['source_results_path']))
T,P=r['input']['total_length'],r['input']['prompt_length'];keep=set(r['input']['keep'])
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(v):return {'net':float(v.sum()),'positive_sum':float(v.clip(min=0).sum()),'negative_sum':float(v.clip(max=0).sum()),'absolute_sum':float(np.abs(v).sum())}
rows={name:z[name].astype(np.float64) for name in p['row_fields']}
assert all(v.shape==(1,16,T) and np.isfinite(v).all() for v in rows.values())
den=rows['raw_Jeffreys_denominator'];dnorm=rows['normalized_direction_contraction'];kap=rows['kappa'];num=rows['numerator'];zero=den==0
assert np.all(den>=0) and np.all(kap[zero]==0) and np.all(rows['osc_g']>=0)
assert np.all(rows['raw_p0_sum']>0) and np.all(rows['raw_p1_sum']>0)
assert int(zero.sum())==r['row_audit']['degenerate_count']
ratio=np.zeros_like(den);np.divide(dnorm,den,out=ratio,where=~zero)
predicted=-num.copy();predicted[~zero]=num[~zero]*(ratio[~zero]-1)
observed=rows['endpoint_positive']+rows['endpoint_negative']-rows['normalized_route_target'];rest=observed-predicted
mass_bound=rows['osc_g']/np.minimum(rows['raw_p0_sum'],rows['raw_p1_sum'])
derived={'diagnostic_Dnorm_over_Draw_zero_at_degenerate':ratio,'denominator_only_secant_residual':predicted,
    'observed_secant_residual':observed,'remaining_FP32_arithmetic_residual':rest,
    'ideal_kappa_bound_excess':np.maximum(np.abs(kap)-rows['osc_g'],0),
    'mass_corrected_kappa_bound':mass_bound,'mass_corrected_kappa_bound_excess':np.maximum(np.abs(kap)-mass_bound,0)}
for name,value in derived.items():close(value,z[name])
for name,value in [('row_FP32_constraint_residual',observed),('denominator_only_residual',predicted),
    ('remaining_FP32_arithmetic_residual',rest),('Dnorm_minus_Draw',dnorm-den),('row_sum',rows['route_row_sum']),
    ('normalization_target_change',rows['normalized_route_target']-rows['raw_route_target'])]:
    for key,v in stats(value).items():close(v,r['row_audit'][name][key])
close(np.abs(observed).max(),r['row_audit']['row_FP32_constraint_max_absolute'])
close(np.abs(rows['route_row_sum']).max(),r['row_audit']['row_sum_max_absolute'])
close(np.abs(rows['direction_row_sum']).max(),r['row_audit']['direction_row_sum_max_absolute'])
for name in ['ideal_kappa_bound_excess','mass_corrected_kappa_bound_excess']:
    close(derived[name].max(),r['row_audit'][name+'_max']);assert int((derived[name]>0).sum())==r['row_audit'][name+'_count']
out={'status':'independent_supported_secant_audit_passed','analyzer_sha256':sha(Path(__file__)),
    'protocol_sha256':sha(D/'protocol.json'),'results_sha256':sha(D/'results.json'),
    'library_sha256_from_hashed_build_receipt':b['library']['sha256'],'build_seconds':b['wall_seconds'],'run_seconds':r['seconds'],
    'candidate_wrapper_seconds':r['candidate_wrapper_seconds_including_prepare_checks_and_sync'],
    'counts':{k:r[k] for k in ['candidate_finite_entered','candidate_finite_returned','native_FA_calls','model_calls','DT_calls','scorer_calls','FT_calls']},
    'dv_bitwise_equal_to_original':True,'row_audit':r['row_audit'],'points':{},
    'FP32_formula_rounding':{'kappa_Draw_minus_numerator_max':float(np.abs(kap*den-num).max()),
        'target_minus_base_minus_numerator_max':float(np.abs(rows['normalized_route_target']-rows['base_gradient_contraction']-num).max())},
    'verification_scope':'Public NPZ/receipt/source algebra independently recomputed. Runtime dv identity uses pinned study and hashed receipt; private coefficients/new .so are not relabeled as locally rehashed tensors.',
    'protected_paths_not_downloaded':[item['path'] for item in p['protected_sources'] if not local(item['path']).is_file()]}
for step,row in r['points'].items():
    v={name:z[step+'_'+name] for name in row['fields']};assert all(x.shape==(1,T) and np.isfinite(x).all() for x in v.values())
    close(v['old_core_error'],v['old_QK']+v['unchanged_V']-v['actual_at_stored_seed'])
    close(v['candidate_core_error'],v['candidate_QK']+v['unchanged_V']-v['actual_at_stored_seed'])
    close(v['candidate_minus_old_error'],v['candidate_QK']-v['old_QK'])
    close(v['candidate_core_error']-v['old_core_error'],v['candidate_minus_old_error'])
    if step=='B2':
        groups={'eligible_prompt':sorted(keep),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
        close(v['normalized_route_target'],rows['normalized_route_target'].sum(1));close(v['raw_route_target'],rows['raw_route_target'].sum(1))
        close(v['candidate_QK_minus_normalized_route_target'],v['candidate_QK']-v['normalized_route_target'])
    else:
        deleted=set(source['points'][step]['input_receipt']['deleted_positions']);assert deleted<=keep
        groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
        close(v['old_QK'],hz[step+'_qk_prediction'])
        close(v['old_route_error_at_V0'],v['old_QK']-hz[step+'_R0']);close(v['candidate_route_error_at_V0'],v['candidate_QK']-hz[step+'_R0'])
        close(v['old_core_error'].sum(),source['points'][step]['layer_decompositions']['19']['terms']['finite_FA_core_including_seed_cast'])
    assert sorted(i for ids in groups.values() for i in ids)==list(range(T))
    for name,x in v.items():
        for key,value in stats(x).items():close(value,row['fields'][name][key])
        for group,ids in groups.items():
            assert row['groups'][group]['count']==len(ids)
            for key,value in stats(x[0,ids]).items():close(value,row['groups'][group]['fields'][name][key])
    out['points'][step]={'fields':row['fields'],'groups':row['groups'],
        'local_core_absolute_error_reduction':abs(float(v['old_core_error'].sum()))-abs(float(v['candidate_core_error'].sum())),
        'coordinate_error_absolute_sum_change':float(np.abs(v['candidate_core_error']).sum()-np.abs(v['old_core_error']).sum())}
out['whole_input_baseline_sign_risk']={k:source['points'][k]['complete_input'] for k in ['1','10','20']}
out['interpretation']='One saved FA19 coefficient change only. Analyze signed net and coordinate cancellation together. Local improvement can support a candidate-specific next decision; it cannot establish full-method MAS/RISE/needle, all-layer generalization, or repair of the failed Euclidean candidate.'
out['precision_scope']='D_raw uses the stable nonnegative unnormalized expression; the normalized Dnorm is observed, not fed back. Their secant residual plus remaining FP32 error are explicitly separate. No exact floating-point conservation claim.'
out['audit_seconds']=time.perf_counter()-started;target=A/'dt_fa19_supported_secant_summary_20260908.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'output':str(target),'sha256':sha(target),'seconds':out['audit_seconds']}))
