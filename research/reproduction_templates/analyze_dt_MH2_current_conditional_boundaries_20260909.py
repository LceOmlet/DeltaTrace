"""Independent NumPy audit of one MH2 DT/four native-score boundary diagnostic."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_current_conditional_boundaries_20260909_v1'
started=time.perf_counter();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
BOUNDARIES=[str(i) for i in range(33)]+['norm']
STEPS=[0,3,10,20]

def close(x,y,label='',atol=1e-7):
    error=float(np.max(np.abs(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)),initial=0))
    assert error<atol,(label,error)
    return error

def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum()),
        'absolute':float(np.abs(x).sum())}

def drift(x,y):
    d=np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    return {'relative_L2':float(np.linalg.norm(d)/max(np.linalg.norm(y),1e-30)),
        'max_absolute':float(np.max(np.abs(d),initial=0)),'bitwise_equal':bool(np.array_equal(x,y))}

p,r=read(D/'protocol.json'),read(D/'results.json')
assert p==r['protocol']==read(A/'dt_MH2_current_conditional_boundaries_protocol_20260909.json')
assert sha(D/'protocol.json')=='b4e342eca0aa85145f6ebb047149517e314bfdec617b16eb557cf1d171e52728'
assert p['files_sha256']['study.py']=='c9304873d0f2e4944c6178058b54f72e7299286870a31c1fcc0d141481d06f54'
assert r['status']=='MH2_current_conditional_boundaries_1DT4score_complete'
receipt=read(D/'terminal_receipt.json')
assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
for name,item in receipt['files'].items():
    want=item if isinstance(item,str) else item['sha256'];assert sha(D/name)==want,name
    if isinstance(item,dict) and 'bytes' in item:assert (D/name).stat().st_size==item['bytes']
for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    assert not any(name.endswith('.pt') for name in archive.namelist())
    for name in archive.namelist():assert archive.read(name)==(D/name).read_bytes(),name
assert r['sources_before']==r['sources_after']
assert r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
assert r['model_loads']==r['native_eager_diagnostics']==r['DT_entered']==r['DT_returned']==1
assert r['scorer_entered']==r['scorer_returned']==4
assert r['FT_entered']==r['FT_calls']==r['generation_calls']==0
assert p['capture_steps']==STEPS and p['call_schedule']==[['morehopqa_2','DT']]
assert r['finite_counts']['DT']=={'entered':8,'returned':8}
fc=r['finite_counts']['FLA_backend'];assert fc['entered']==fc['returned']==25
assert fc['native_adjoint_stages_from_returned_calls']==50 and fc['native_stages_inside_nonreturned_calls']==0
wrapper=r['finite_counts']['layer0_average_wrapper'];assert wrapper['entered']==wrapper['returned']==1
assert len(wrapper['calls'])==1 and wrapper['calls'][0]['status']=='returned'
assert all(call['status']=='returned' and call['seconds']>=0 for call in r['calls'])
assert sha(D/'vectors.npz')==r['vectors_sha256']
source=p['source_standalone'];S=A/'snapshot'/Path(source['results_path']).as_posix().lstrip('/')
S=S.parent
for kind in ['results','vectors','protocol']:assert sha(S/(kind+('.npz' if kind=='vectors' else '.json')))==source[kind+'_sha256']
prior=read(S/'results.json');source_case=prior['cases']['morehopqa_2'];curve=source_case['curves'][p['source_method']]
sv=np.load(S/'vectors.npz',allow_pickle=False);v=np.load(D/'vectors.npz',allow_pickle=False)
case=r['cases']['morehopqa_2'];info=case['input'];T,P=info['total_length'],info['prompt_length'];keep=set(info['keep'])
assert info==source_case['input'] and T==710 and P==480
for key,new_freeze in r['input_freeze_before_model_load'].items():
    old_freeze=prior['input_freeze_before_model_load'][key]
    for field in ['input','input_ids','target_ids','baseline_sha256']:assert new_freeze[field]==old_freeze[field],(key,field)
ids=np.asarray(r['input_freeze_before_model_load']['morehopqa_2']['input_ids'],dtype=np.int64);eos=int(ids[-1])
assert hashlib.sha256(ids.tobytes()).hexdigest()==info['input_sha256']
mask=r['source_mask_closure'];assert mask['all_21_receipts_reconstructed'] and mask['source_vector_descending_on_frozen_order'] and mask['new_vector_cannot_change_masks']
assert mask['sorted_keep']==curve['sorted_keep'] and sorted(mask['sorted_keep'])==sorted(keep)
assert mask['capture_receipts']==p['frozen_capture_receipts']=={str(step):curve['input_receipts'][step] for step in STEPS}
seen=set();x=ids.copy();offset=0;n,extra=divmod(len(keep),20)
for step,rec in enumerate(curve['input_receipts']):
    if step:
        g=curve['sorted_keep'][offset:offset+n+(step-1<extra)];offset+=len(g)
        assert g==mask['groups'][step-1];seen.update(g);x[g]=eos
    assert rec['deleted_positions']==sorted(seen) and np.where(x!=ids)[0].tolist()==sorted(seen)
    assert hashlib.sha256(x.tobytes()).hexdigest()==rec['input_sha256']
assert seen==keep
assert len(r['runs'])==1;run=r['runs'][0];details=run['details']
assert run['status']=='complete' and run['root_forwards']==1
assert details['norm_gate_rules']=={'0':'symmetric'} and details['finite_fla_by_layer']==[0]
assert run['finite_callback_counts']=={'FA_entered':8,'FA_returned':8,'FLA_backend_entered':25,'FLA_backend_returned':25}
assert len(details['layers'])==32 and list(details['layers'])==[str(i) for i in reversed(range(32))]
assert set(run['boundary_coefficient_shapes'])==set(BOUNDARIES)
shapes=list(run['boundary_coefficient_shapes'].values());assert all(s==shapes[0] for s in shapes) and shapes[0][:2]==[1,T]
kinds=[c['kind'] for c in details['calls']]
assert [k for k in kinds if k.startswith('native_replay_')]==['native_replay_'+str(i) for i in reversed(range(32))]
assert [k for k in kinds if k.startswith('finite_decoder_')]==['finite_decoder_'+str(i) for i in reversed(range(32))]
assert sum(k.startswith('public_FA_LSE_') for k in kinds)==8
for row in details['layers'].values():
    assert row['decoder_calls']=={k:1 for k in ['input_norm','post_norm','gate','up','silu','down','mlp','decoder']}
    assert row['mixer_calls']==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if row['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
w=v['morehopqa_2_DT_evaluated'];full=v['morehopqa_2_DT_full'];old=sv['morehopqa_2_'+p['source_method']+'_evaluated']
assert w.dtype==np.float32 and w.shape==(P,) and full.shape==(T,) and np.isfinite(full).all()
assert np.array_equal(full[:P].astype(np.float32),w)
close(full.sum(),details['signed_sum'],'saved full sum')
vector_drift=drift(full,sv['morehopqa_2_'+p['source_method']+'_full'])
for key,value in vector_drift.items():close(value,run['source_vector_drift_report_only'][key],key,1e-12)
close(run['source_B2_root_effect'],prior['runs'][p['source_run_index']]['details']['root_effect'])
close(run['current_minus_source_B2_root_effect'],details['root_effect']-run['source_B2_root_effect'])
assert set(case['points'])=={str(i) for i in STEPS}
for step in STEPS:
    point=case['points'][str(step)]
    assert point['status']=='complete' and point['input_receipt']==curve['input_receipts'][step]
    assert point['boundary_capture_counts']=={b:1 for b in BOUNDARIES}
    close(point['source_original_native_score'],curve['scores'][step])
    close(point['current_minus_source_native_score'],point['original_native_score']-curve['scores'][step])
expected_keys={'morehopqa_2_DT_evaluated','morehopqa_2_DT_full'}
for step in [3,10,20]:
    expected_keys.update('step'+str(step)+'_boundary_'+name for name in BOUNDARIES)
    expected_keys.update('step'+str(step)+'_decoder_'+str(i)+'_actual_minus_predicted' for i in range(32))
assert set(v.files)==expected_keys
out={'status':'MH2_current_conditional_boundaries_independent_CPU_audit_passed',
    'next':None,'analyzer_sha256':sha(Path(__file__)),'protocol_sha256':sha(D/'protocol.json'),
    'results_sha256':sha(D/'results.json'),'vectors_sha256':sha(D/'vectors.npz'),'source_standalone':source,
    'input':{k:z for k,z in info.items() if k!='keep'},'sign_convention':case['sign_convention'],
    'actual_budget':{'complete_DT_entered':r['DT_entered'],'complete_DT_returned':r['DT_returned'],
        'native_B1_scorer_entered':r['scorer_entered'],'native_B1_scorer_returned':r['scorer_returned'],
        'model_loads':1,'native_eager_NI0_initialization':1,'native_DT_B2_roots':1,'native_decoder_replays':32,
        'finite_decoder_calls':32,'finite_FA_calls':8,'auxiliary_FA_calls':8,'finite_FLA_backend_calls':25,
        'native_FLA_adjoint_stages':50,'FT_calls':0,'generation_calls':0,'CPU_audit_model_calls':0,
        'job_seconds':r['seconds'],'DT_with_observer_seconds':run['outer_attribute_seconds'],
        'four_scorer_with_observer_seconds':sum(c['seconds'] for c in r['calls'] if c['kind'].startswith('original_scorer_')),
        'private_save_seconds':sum(c['seconds'] for c in r['calls'] if c['kind']=='save_actual_private_boundary_tensors'),
        'failed_entered_minus_returned':{'DT':r['DT_entered']-r['DT_returned'],'scorer':r['scorer_entered']-r['scorer_returned']},
        'scope':'This diagnostic job only; all observer copying/contraction overhead included, not production latency.'},
    'current_source_full_vector_drift':vector_drift,'current_source_evaluated_vector_drift':drift(w,old),
    'current_source_B2_root_effect_difference':run['current_minus_source_B2_root_effect'],
    'actual_source_mask_native_scores':{str(s):case['points'][str(s)] for s in STEPS},
    'B1_allEOS_minus_B2_boundary_effect':case['B1_allEOS_minus_B2_boundary_effect'],
    'private_artifact':case['private_artifact'],'steps':{},
    'proof_scope':'Local NumPy verifies the saved per-token contractions, every decoder difference and scalar telescoping, receipts and source/weight identities. Large private actual coefficient/activation tensors remain remote and were not rehashed or dot-product recomputed locally. No new model forward, metric curve or candidate claim.'}
for step in [3,10,20]:
    ss=str(step);saved=case['decomposition'][ss];points={b:v['step'+ss+'_boundary_'+b] for b in BOUNDARIES}
    for name,values in points.items():
        assert values.shape==(T,) and values.dtype==np.float64 and np.isfinite(values).all()
        close(values.sum(),case['B1_boundary_contractions'][ss][name],ss+' boundary '+name)
    jumps={i:v['step'+ss+'_decoder_'+str(i)+'_actual_minus_predicted'] for i in range(32)}
    token_error=0
    for i,values in jumps.items():
        token_error=max(token_error,close(values,points[str(i+1)]-points[str(i)],ss+' decoder '+str(i),1e-12))
        close(values.sum(),saved['decoder_errors'][str(i)])
    close(sum(jumps.values()),points['32']-points['0'],ss+' token telescope')
    deleted=curve['input_receipts'][step]['deleted_positions'];pred=float(w[deleted].astype(np.float64).sum())
    prior_pred=float(old[deleted].astype(np.float64).sum());actual=case['points']['0']['original_native_score']-case['points'][ss]['original_native_score']
    actual32=case['points']['0']['FP32_same_native_logits_diagnostic']-case['points'][ss]['FP32_same_native_logits_diagnostic']
    terms={'native_BF16_score_minus_same_logits_FP32':actual-actual32,
        'head_and_logprob_seed':actual32-points['norm'].sum(),'final_norm':points['norm'].sum()-points['32'].sum(),
        'decoders':sum(float(z.sum()) for z in jumps.values()),'input_map':points['0'].sum()-pred}
    for name,value in terms.items():close(value,saved['terms'][name],ss+' term '+name)
    close(sum(terms.values()),actual-pred,ss+' scalar telescope')
    for name,value in [('actual_native_logprob_drop',actual),('same_native_logits_FP32_drop',actual32),
        ('current_DT_predicted_drop',pred),('actual_minus_predicted',actual-pred),('source_DT_predicted_drop',prior_pred)]:close(value,saved[name],ss+' '+name)
    old_actual=curve['scores'][0]-curve['scores'][step]
    close(saved['source_actual_logprob_drop'],old_actual);close(saved['source_actual_minus_predicted'],old_actual-prior_pred)
    score_shift=actual-old_actual;vector_shift=-(pred-prior_pred)
    close(saved['conditional_error_drift_from_new_score'],score_shift)
    close(saved['conditional_error_drift_from_new_vector'],vector_shift)
    close((actual-pred)-(old_actual-prior_pred),score_shift+vector_shift)
    layer_sums=np.asarray([jumps[i].sum() for i in range(32)]);ordered=sorted(range(32),key=lambda i:abs(layer_sums[i]),reverse=True)
    assert [int(pair[0]) for pair in saved['decoder_errors_ranked_absolute']]==ordered
    assert int(saved['most_overpredicting_decoder'])==int(np.argmin(layer_sums))
    close(layer_sums[p['expected_FA_layers']].sum(),saved['FA_decoder_sum'])
    close(layer_sums[p['expected_GDN_reverse_order']].sum(),saved['GDN_decoder_sum'])
    groups={'deleted_prompt':deleted,'kept_eligible_prompt':sorted(keep-set(deleted)),
        'other_prompt':sorted(set(range(P))-keep),'fixed_response':list(range(P,T))}
    rankings=[]
    for i in ordered:
        rankings.append({'layer':i,'kind':details['layers'][str(i)]['block_type'],
            'jump':float(layer_sums[i]),'token_signed_mass':stats(jumps[i]),
            'groups':{k:dict(count=len(indices),**stats(jumps[i][indices])) for k,indices in groups.items()}})
    out['steps'][ss]={'actual_logprob_drop':actual,'predicted_drop':pred,'actual_minus_predicted':actual-pred,
        'source_actual_minus_predicted':old_actual-prior_pred,'error_drift_from_score':score_shift,'error_drift_from_vector':vector_shift,
        'terms':{k:float(z) for k,z in terms.items()},'decoder_layer_signed_mass':stats(layer_sums),
        'all_decoder_token_signed_mass':stats(np.stack(list(jumps.values()))),
        'largest_overprediction_layer':int(np.argmin(layer_sums)),'largest_compensating_layer':int(np.argmax(layer_sums)),
        'decoder_ranking_absolute':rankings,'FA_decoder_sum':float(layer_sums[p['expected_FA_layers']].sum()),
        'GDN_decoder_sum':float(layer_sums[p['expected_GDN_reverse_order']].sum()),'per_token_closure_max_absolute':token_error,
        'scalar_closure_absolute':float(abs(sum(terms.values())-(actual-pred)))}
    if step==20:
        for name in BOUNDARIES:close(points[name].sum()-run['B2_endpoint_boundary_contractions'][name],case['B1_allEOS_minus_B2_boundary_effect'][name])
private=case['private_artifact'];assert private['bytes']>0 and private['tensor_CPU_bytes']>0
assert private['file']=='MH2_actual_boundary_private.pt' and len(private['sha256'])==64
mid=out['steps']['10'];early=out['steps']['3'];layer=mid['largest_overprediction_layer']
out['next']={'status':'internal_operator_measurement_needed_after_root_review',
    'predeclared_selection':'Largest negative mid-step decoder jump on the fixed current-control mask; early and allEOS retained as controls.',
    'mid_layer':layer,'early_layer':early['largest_overprediction_layer'],'kind':details['layers'][str(layer)]['block_type'],
    'reusable_remote_private_path':'/tmp/'+D.name+'/'+private['file'],
    'minimum_missing_measurement':'Use saved current-run upstream/input coefficients and actual native boundary inputs for the selected decoder. Capture that decoder native internal features for saved B2 and B1 clean/step3/step10/allEOS, then one finite decoder pullback with fixed current rule and retained upstream; partition MLP, residual/RMSNorm, mixer and native-rounding terms. Check replay/native-transfer drift explicitly. No complete attribution or scorer repeat.',
    'candidate_not_authorized':True,'no_new_model_call_from_this_analyzer':True,
    'budget_to_freeze_if_authorized':{'complete_DT':0,'whole_model_forward':0,'original_scorer':0,'FT':0,'generation':0,
        'selected_native_decoder_replays':5,'selected_finite_decoder':1,
        'boundary_kwargs_note':'Selected-layer native rotary/cache/mask arguments must be sourced/verified against the original saved input. Current private file saves boundaries but no internal features or replay kwargs; do not silently invent a different layer forward.'}}
out['limits']=['Conditional boundary jumps localize disagreement for fixed coefficients and observed deletion inputs; a large jump is not proof that one internal operator is the causal root or that a repair will improve MAS.',
    'AllEOS closure and compensation cannot establish faithfulness for arbitrary partial deletions. Source/current drift is separately measured, not automatically classified as error or waived.',
    'No new original MAS, RISE, FT comparison, whole-benchmark conclusion or production speed claim is produced.']
out['CPU_audit_seconds']=time.perf_counter()-started
path=A/'dt_MH2_current_conditional_boundaries_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'summary':str(path),'summary_sha256':sha(path),
    'steps':{k:{name:row[name] for name in ['actual_minus_predicted','largest_overprediction_layer','largest_compensating_layer','FA_decoder_sum','GDN_decoder_sum']} for k,row in out['steps'].items()},'next':out['next'],'seconds':out['CPU_audit_seconds']}))
