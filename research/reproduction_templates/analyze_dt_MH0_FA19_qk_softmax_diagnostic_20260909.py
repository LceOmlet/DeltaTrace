"""Independent NumPy MH0 QK/softmax ledger and existing query-row concentration audit."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_qk_softmax_diagnostic_20260909_v1'
started=time.perf_counter();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes());local=lambda remote:A/'snapshot'/remote.lstrip('/')
def close(x,y,label='',tol=1e-7):
    err=float(np.max(np.abs(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)),initial=0));assert err<tol,(label,err)
    return err
def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return {'net':float(x.sum()),'positive_sum':float(x.clip(min=0).sum()),'negative_sum':float(x.clip(max=0).sum()),'absolute_sum':float(np.abs(x).sum()),'max_absolute':float(np.max(np.abs(x),initial=0))}
p,r=read(D/'protocol.json'),read(D/'results.json');receipt=read(D/'terminal_receipt.json')
assert p==r['protocol']==read(A/'dt_MH0_FA19_qk_softmax_diagnostic_protocol_20260909.json')
assert sha(D/'protocol.json')=='b94e27dfa21bf63046e8b23bcaa18e024549e7a77076893094bafe236039b1f1'
assert r['status']=='MH0_three_FA19_conditional_score_contractions_complete'
assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
for name,item in receipt['files'].items():
    assert sha(D/name)==(item if isinstance(item,str) else item['sha256']),name
    if isinstance(item,dict) and 'bytes' in item:assert (D/name).stat().st_size==item['bytes']
for name,want in p['files_sha256'].items():assert sha(D/name)==want
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    assert not any(n.endswith('.pt') for n in archive.namelist())
    for name in archive.namelist():assert archive.read(name)==(D/name).read_bytes(),name
not_local=[]
for item in p['protected_sources']:
    path=local(item['path'])
    if path.is_file():assert sha(path)==item['sha256'],item['path']
    else:
        assert item['path'] in [p['reused_build']['library_path'],p['production_library_path']],item['path']
        not_local.append(item)
b=read(local(p['reused_build']['build_results_path']));bp=read(local(p['reused_build']['protocol_path']))
assert b['status']=='finite_extension_compiled_not_executed' and b['compile_returncode']==0 and b['protocol']==bp
assert b['vendor_sources_before']==b['vendor_sources_after']
assert sha(local(p['reused_build']['build_results_path']))==r['build_results_sha256']==p['reused_build']['build_results_sha256']
assert b['library']['sha256']==r['diagnostic_library_sha256']==p['reused_build']['library_sha256']
assert r['reused_build']==p['reused_build'] and r['compiler_attempts']==0
assert r['native_FA_entered']==r['native_FA_returned']==1
assert r['diagnostic_finite_entered']==r['diagnostic_finite_returned']==3
assert r['diagnostic_phase_launches_expected']==r['diagnostic_phase_launches_from_returned']==9 and r['phases_inside_nonreturned_diagnostic']==0
assert all(r[k]==0 for k in ['model_calls','DT_calls','scorer_calls','FT_calls','backward_calls','generation_calls'])
h=read(local(p['hybrid_results_path']));source=read(local(p['source_results_path']))
assert h['status']=='MH0_eleven_native_FA_hybrid_contrasts_complete' and source['status']=='MH0_current_FA19_1native_kwargs5replay1finite_internal_complete'
assert r['input']==source['input'] and r['input']['total_length']==853 and p['fixed_steps']==['3','10','20']
assert sha(D/'conditional_signed_rows.npz')==r['artifacts']['conditional_signed_rows.npz']['sha256']
z=np.load(D/'conditional_signed_rows.npz',allow_pickle=False);hz=np.load(local(p['hybrid_vectors_path']),allow_pickle=False)
terms=['coefficient_replay_difference','QK_interaction_and_multiplier_rounding','tile_weight_BF16_cast','finite_softmax_condition_and_native_precision']
P,T=r['input']['prompt_length'],r['input']['total_length'];keep=set(r['input']['keep'])
boundary=read(local(source['protocol']['source_boundary']['results_path']));ids=boundary['input_freeze_before_model_load']['morehopqa_0']['input_ids']
out={'status':'MH0_FA19_qk_softmax_independent_CPU_and_query_row_audit_passed',
    'next':'No new GPU or candidate from this audit. Retain the larger measured PV-reference interaction while root evaluates existing endpoint-allocation options.',
    'analyzer_sha256':sha(Path(__file__)),'protocol_sha256':sha(D/'protocol.json'),'results_sha256':sha(D/'results.json'),
    'conditional_signed_rows_sha256':sha(D/'conditional_signed_rows.npz'),'build_results_sha256':r['build_results_sha256'],
    'diagnostic_library_sha256':r['diagnostic_library_sha256'],
    'actual_budget':{'compiler_attempts':0,'native_public_B2_FA':1,'diagnostic_finite_entered':3,'diagnostic_finite_returned':3,'diagnostic_phases_from_returns':9,
        'model':0,'DT':0,'scorer':0,'FT':0,'backward':0,'generation':0,'job_seconds':r['seconds'],'CPU_audit_model_calls':0,
        'failed_entered_minus_returned':0,'historical_compile_seconds_not_new_cost':b['wall_seconds'],
        'scope':'This existing-library diagnostic only. Old build time is provenance and is not charged as a new compilation.'},
    'public_B2_replay_drift':r['public_B2_replay_drift'],'saved_B1_vs_B2_endpoint_drift':r['saved_B1_vs_B2_endpoint_drift'],
    'input':{k:v for k,v in r['input'].items() if k!='keep'},'points':{},
    'proof_scope':'Local NPZ verifies every saved token-row identity/group and its current native-hybrid reference. Existing library identity follows matched build/execution SHA receipts; unavailable local binaries and private tensor operands are not claimed to be locally rehashed.',
    'protected_binaries_verified_by_remote_receipts_only':not_local}
rows={};allkeys=set()
for step in p['fixed_steps']:
    row=r['points'][step];hr=h['points'][step];fields={n:z[step+'_'+n] for n in row['fields']};allkeys.update(step+'_'+n for n in fields)
    assert row['input_receipt']==hr['input_receipt']==p['frozen_input_receipts'][step]
    assert all(row['coefficient_independent_of_actual_A'].values())
    assert all(x.shape==(1,T) and x.dtype==np.float64 and np.isfinite(x).all() for x in fields.values())
    close(fields['saved_qk_prediction'],hz[step+'_qk_prediction']);close(fields['R0'],hz[step+'_R0'])
    for term,left,right in zip(terms,['saved_qk_prediction','replayed_qk_prediction','T_BF16','T_FP32'],['replayed_qk_prediction','T_BF16','T_FP32','R0']):close(fields[term],fields[left]-fields[right],term)
    maxclosure=close(sum(fields[k] for k in terms),fields['saved_qk_prediction']-fields['R0'],'four term token closure')
    route=float((fields['saved_qk_prediction']-fields['R0']).sum());close(route,row['original_route_prediction_error']);close(route,p['original_route_error'][step]);close(route,hr['fields']['routing_at_baseline_values_prediction_error']['net'])
    deleted=set(row['input_receipt']['deleted_positions']);assert deleted<=keep
    groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
    assert sorted(i for values in groups.values() for i in values)==list(range(T))
    for name,x in fields.items():
        for key in row['fields'][name]:close(stats(x)[key],row['fields'][name][key])
        for group,indices in groups.items():
            assert row['groups'][group]['count']==len(indices)
            for key in row['groups'][group]['fields'][name]:close(stats(x[:,indices])[key],row['groups'][group]['fields'][name][key])
    seed=hr['BF16_seed_contrast_differences']['R0'];e=fields[terms[-1]][0];tfp=fields['T_FP32'][0];r0=fields['R0'][0];rows[step]=e
    order=np.argsort(-np.abs(e),kind='stable');absolute=float(np.abs(e).sum());positive=float(e.clip(min=0).sum())
    top=[]
    for i in order[:20]:
        top.append({'query_input_position_zero_based':int(i),'role':'fixed_response' if i>=P else 'prompt',
            'response_input_offset_zero_based':int(i-P) if i>=P else None,'input_token_id':ids[int(i)],
            'T_FP32_fixed_pair_response':float(tfp[i]),'R0_native_routing_response':float(r0[i]),'softmax_condition_native_residual':float(e[i])})
    concentration={str(k):{'absolute_fraction':float(np.abs(e[order[:k]]).sum()/absolute) if absolute else None,
        'signed_sum':float(e[order[:k]].sum()),'positive_fraction':float(e[order[:k]].clip(min=0).sum()/positive) if positive else None} for k in [1,5,10,20,40,100]}
    selected_window=list(range(430,448))
    out['points'][step]={'original_route_error':route,'four_terms':{k:stats(fields[k]) for k in terms},
        'actual_contractions':{k:stats(fields[k]) for k in ['saved_qk_prediction','T_BF16','T_FP32','R0']},
        'coefficient_replay_drift':row['coefficient_replay_drift'],'max_token_algebra_closure':maxclosure,
        'BF16_minus_FP32_seed_R0_effect_from_hybrid':seed,
        'softmax_term_if_R0_uses_BF16_seed':float(e.sum()-seed['net']),
        'larger_PV_reference_interaction':hr['fields']['routing_content_interaction_contrast'],
        'query_error_groups':{group:dict(count=len(indices),**stats(e[indices])) for group,indices in groups.items()},
        'query_absolute_fraction_in_fixed_response':float(np.abs(e[P:]).sum()/absolute),
        'query_top_by_absolute_error':top,'query_concentration':concentration,
        'descriptive_shared_window_430_to_447':dict(count=len(selected_window),**stats(e[selected_window]),selection_scope='Descriptive window covering several observed top query rows, chosen after inspection; not an independent test.'),
        'coordinates':'Residual T_FP32-R0 is at output-query positions after summing heads; Q/K predicted contractions use operand positions. These are not original source-token attribution errors.'}
assert set(z.files)==allkeys
early,mid=rows['3'],rows['10'];order3=np.argsort(-np.abs(early),kind='stable');order10=np.argsort(-np.abs(mid),kind='stable')
out['early_mid_query_pattern']={'cosine':float(np.dot(early,mid)/(np.linalg.norm(early)*np.linalg.norm(mid))),
    'top10_absolute_query_overlap':sorted(set(map(int,order3[:10]))&set(map(int,order10[:10]))),
    'scope':'Two fixed deletion conditions of the same example; no independent generalization claim.'}
out['logmean_directional_interpretation']={
    'source_extension_sha256':p['files_sha256']['vendor_fa_finite_p1_bf16_d256_conditional_diag_20260908.cu'],
    'fixed_weight_formula':'For each valid causal row, l_j=L(p0_j,p1_j), c=sum(l_j*g_j)/sum(l_j), W_j=l_j*(g_j-c), where g_j is the retained BF16 upstream/value0 score contraction. T_FP32=sum_j W_j*(s_Cj-s_Aj) using the pinned diagnostic score MMA; R0 is the unchanged public native FA routing contrast at retained V0.',
    'exact_arithmetic_limit':'With exactly normalized positive endpoint probabilities, W is zero-row-sum and conserves the original EOS/clean score chord. It is a logmean weighted secant rule, not demonstrated to be the Jacobian integral along the actual partial-deletion chord. Endpoint conservation does not force T=R0 on other logit directions.',
    'observed_conclusion':'The frozen EOS/clean weights produce a substantially different scalar response to the observed C-minus-A score direction than the native routing contrast. This is a measured conditional directional mismatch. The largest positive rows include both excessive positive predicted response and missing negative native response; a single global amplification factor is not established.',
    'unresolved':'No per-key score or probability rows are exported, so these data do not determine chord angles, which key probabilities drive the discrepancy, or whether any individual logmean weight is excessive. The diagnostic uses FP32 MMA/reductions with BF16 operands and native LSE; native output and B1/B2 transfer remain in the named residual.',
    'head_identifiability':'Public T and native hybrid R0 are head-summed. Per-head R0 was not retained. Existing per-head T alone cannot identify head-specific residuals; no head root-cause claim and no extra FA/export requested.'}
out['precision_limits']='Zero measured coefficient/B2 replay drift establishes equality of those saved tensors in these executions. Small total BF16 tile/seed terms do not imply every token, head, score, probability or reduction is numerically exact. Signed cancellation and per-row absolute cast magnitudes are retained; no precision repair follows.'
out['interpretation']='Softmax-condition/native-precision term dominates the routing-at-V0 component. Most absolute residual lies at fixed-response query positions, with a recurring but non-single-row concentration. PV-reference interaction remains larger than routing-at-V0 on both early and mid conditions and must not be ignored. No candidate, new GPU call or metric result is produced.'
out['CPU_audit_seconds']=time.perf_counter()-started
path=A/'dt_MH0_FA19_qk_softmax_diagnostic_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'summary':str(path),'sha256':sha(path),'points':{s:{'terms':{k:v['net'] for k,v in q['four_terms'].items()},'response_abs_fraction':q['query_absolute_fraction_in_fixed_response'],'top20_abs_fraction':q['query_concentration']['20']['absolute_fraction'],'top_queries':q['query_top_by_absolute_error'][:3]} for s,q in out['points'].items()},'pattern':out['early_mid_query_pattern'],'seconds':out['CPU_audit_seconds']}))
