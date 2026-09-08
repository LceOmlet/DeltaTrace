"""Bounded independent CPU recomputation of four original metric curves."""
import hashlib,json,math,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1';started=time.perf_counter()
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();read=lambda x:json.loads(x.read_bytes());local=lambda x:A/'snapshot'/x.lstrip('/')
p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
assert r['status']=='layer0_FLA_endpoint_average_stability_cost_10DT245FLA84score_complete' and r['protocol']==p
assert p==read(A/'dt_layer0_fla_endpoint_average_stability_cost_protocol_20260909.json')
for name,v in receipt['files'].items():assert sha(D/name)==v['sha256'] and (D/name).stat().st_size==v['bytes'],name
for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    for name in archive.namelist():assert archive.read(name)==(D/name).read_bytes(),name
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
for name,want in p['official_source_blob_sha1'].items():
    raw=(local(p['official_root'])/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
assert r['model_loads']==r['native_eager_diagnostics']==1 and r['DT_entered']==r['DT_returned']==10
assert r['scorer_entered']==r['scorer_returned']==84 and r['FT_calls']==r['generation_calls']==0
assert all(r['finite_counts'][m]=={'entered':40,'returned':40} for m in ['control','candidate'])
assert r['finite_counts']['FLA_backend']=={'entered':245,'returned':245,'native_adjoint_stages_from_returned_calls':490,'native_stages_inside_nonreturned_calls':0}
wrapped=r['finite_counts']['layer0_average_wrapper']
assert wrapped['entered']==wrapped['returned']==5 and len(wrapped['calls'])==5
assert len(r['runs'])==10 and [[x['case'],x['method'],x['phase']] for x in r['runs']]==p['call_schedule']
for row in wrapped['calls']:
    assert row['status']=='returned' and row['endpoint_permutation']==[1,0]
    assert row['do_dtype']=='torch.bfloat16'
    assert [x['orientation'] for x in row['backend_calls']]==['original','swapped']
    assert all(x['status']=='returned' for x in row['backend_calls'])
assert sha(D/'vectors.npz')==r['vectors_sha256'];vectors=np.load(D/'vectors.npz')
def close(a,b,tol=1e-9):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<tol
def stats(x):return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum()),'absolute':float(np.abs(x).sum())}
def auc(x):return float((x.sum()-x[0]/2-x[-1]/2)/(len(x)-1))
def metric(scores,density):
    response=np.minimum.accumulate(np.clip((scores-scores[-1])/abs(scores[0]-scores[-1]),0,1))
    penalty=np.abs(response-density);corrected=np.clip(response+penalty,0,1);span=corrected.max()-corrected.min()
    corrected=(corrected-corrected.min())/span if span else np.linspace(1,0,len(scores))
    return response,penalty,corrected,[auc(response),auc(corrected),auc(response+penalty)]
def gamma(n):return (n*2**-24)/(1-n*2**-24)
out={'status':'independent_whole_pilot_audit_passed','analyzer_sha256':sha(Path(__file__)),'results_sha256':sha(D/'results.json'),
    'protocol_sha256':sha(D/'protocol.json'),'metric_source_blob_sha1':p['official_source_blob_sha1']['flashtrace/improved.py'],
    'metric_recomputation':'Exact original NumPy response/clipping/monotonicity/alignment/corrected-AUC equations; captured density independently checked against own FP32 weights within analytical FP32 summation bounds.',
    'counts':{'models':1,'eager_initializations':1,'DT':10,'decoder_replays':320,'finite_decoders':320,'auxiliary_FA':80,'finite_FA':80,'GDN_callback_sites':240,'finite_FLA_backend_calls':245,'native_FLA_adjoint_stages':490,'layer0_average_wrapper':5,'original_scores':84,'FT':0,'generation':0},
    'wall_seconds':r['seconds'],'cases':{},'runs':[]}
prior_path=local(p['prior_wholepilot']['results_path']);assert sha(prior_path)==p['prior_wholepilot']['results_sha256']
prior=read(prior_path);old_vectors_path=prior_path.parent/'vectors.npz';assert sha(old_vectors_path)==prior['vectors_sha256']
old_vectors=np.load(old_vectors_path,allow_pickle=False)
for name,want in p['span_source_sha256'].items():
    raw=(local(p['author_data_root'])/name).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==want,name
out['input_provenance']={}
for dataset,index in p['case_indices']:
    key=f'{dataset}_{index}';case=r['cases'][key];frozen=r['input_freeze_before_model_load'][key];info=case['input'];record=p['fixed_records'][key]
    raw=local(p['cache_paths'][dataset]).read_bytes();assert hashlib.sha256(raw).hexdigest()==p['cache_hashes'][dataset]
    line=raw.decode().splitlines()[index];rec=json.loads(line)
    assert hashlib.sha256(line.encode()).hexdigest()==record['source_record_sha256']==case['mapping']['source_record_sha256']
    assert hashlib.sha256(rec['prompt'].encode()).hexdigest()==record['prompt_text_sha256']
    assert hashlib.sha256(rec['target'].encode()).hexdigest()==record['fixed_target_text_sha256']
    assert frozen['input']==info and frozen['mapping']==case['mapping']
    assert case['mapping']['original_cached_spans_reproduced'] is True
    assert case['mapping']['original_sink_span']==rec['sink_span']
    assert case['mapping']['gold_function']=='unchanged author ruler_gold_prompt_token_indices'
    assert case['gold']==case['mapping']['gold']
    if record['expected_input'] is not None:assert info==record['expected_input']
    if record['expected_gold'] is not None:assert case['gold']==record['expected_gold']
    if not rec['metadata'].get('needle_spans'):assert case['gold']==[]
    ids=np.asarray(frozen['input_ids'],dtype=np.int64);target=np.asarray(frozen['target_ids'],dtype=np.int64)
    assert ids.shape==(info['total_length'],) and target.shape==(info['target_length'],)
    assert np.array_equal(ids[info['prompt_length']:],target)
    assert hashlib.sha256(ids.tobytes()).hexdigest()==info['input_sha256']
    base=ids.copy();base[info['keep']]=target[-1]
    assert hashlib.sha256(base.tobytes()).hexdigest()==frozen['baseline_sha256']==case['baseline_sha256']
    assert np.where(base!=ids)[0].tolist()==info['keep']
    out['input_provenance'][key]={'source_record_sha256':record['source_record_sha256'],'input_sha256':info['input_sha256'],
        'total_length':info['total_length'],'gold':case['gold'],'mapping':case['mapping'],
        'scope':'Author record and original span/gold function source hashes verified. Actual saved input IDs, target suffix, EOS baseline and original input hashes independently reconstructed; tokenizer is not reexecuted locally.'}

def needle_audit(w,keep,gold,reported):
    gold=set(gold)&set(keep)
    if not gold:assert reported is None;return None
    ww=np.maximum(w[keep],0);top=max(1,math.ceil(len(keep)*.1));threshold=np.sort(ww)[-top]
    above={keep[i] for i in np.where(ww>threshold)[0]};ties={keep[i] for i in np.where(ww==threshold)[0]};slots=top-len(above)
    lo=(len(above&gold)+max(0,slots-len(ties-gold)))/len(gold);hi=(len(above&gold)+min(slots,len(ties&gold)))/len(gold)
    assert lo-1e-12<=reported<=hi+1e-12
    return {'reported':reported,'independent_tie_aware_interval':[lo,hi],'gold_denominator':len(gold)}

def drift(now,reference):
    delta=now.astype(np.float64)-reference.astype(np.float64);den=float(np.linalg.norm(reference.astype(np.float64)))
    return {'bitwise_equal':bool(np.array_equal(now,reference)),'relative_L2':float(np.linalg.norm(delta)/den) if den else None,
        'max_absolute':float(np.abs(delta).max(initial=0)),'signed_sum_difference':float(delta.sum())}

for run in r['runs']:
    assert run['status']=='complete' and run['root_forwards']==1
    candidate=int(run['method']=='candidate')
    counts=run['counts']
    assert {k:counts[k] for k in ['native_root','native_decoder_replays','finite_decoder_calls','public_FA_auxiliary_calls','finite_FA_calls','finite_FLA_calls']}=={'native_root':1,'native_decoder_replays':32,'finite_decoder_calls':32,'public_FA_auxiliary_calls':8,'finite_FA_calls':8,'finite_FLA_calls':24}
    assert counts['finite_FLA_backend_calls']==24+candidate
    assert run['finite_callback_counts']=={'FA_entered':8,'FA_returned':8,'FLA_backend_entered':24+candidate,'FLA_backend_returned':24+candidate}
    assert run['native_stage_accounting']=={'returned_FLA_backend_calls_times_two':2*(24+candidate),'stages_inside_nonreturned_backend':0}
    d=run['details'];assert d['norm_gate_rules']=={'0':'symmetric'} and d['finite_fla_by_layer']==([0] if candidate else [])
    assert d['select_output_rows'] is True
    assert len(run['candidate_layer0_receipts'])==candidate
    if candidate:assert run['candidate_layer0_receipts'][0] in wrapped['calls']
    kinds=[x['kind'] for x in d['calls']]
    for prefix in ['native_replay_','finite_decoder_']:
        assert [x for x in kinds if x.startswith(prefix)]==[prefix+str(i) for i in reversed(range(32))]
    assert [x for x in kinds if x.startswith('public_FA_LSE_')]==['public_FA_LSE_'+str(i) for i in reversed(p['expected_FA_layers'])]
    assert list(d['layers'])==[str(i) for i in reversed(range(32))]
    for layer in d['layers'].values():
        assert layer['decoder_calls']=={k:1 for k in ['input_norm','post_norm','gate','silu','up','down','mlp','decoder']}
        assert layer['mixer_calls']==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if layer['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
    case=r['cases'][run['case']];info=case['input'];vk=run['vector_key'];full=vectors[vk+'_full'];w=vectors[vk+'_evaluated']
    assert full.shape==(info['total_length'],) and w.shape==(info['prompt_length'],) and np.array_equal(full[:len(w)].astype(np.float32),w)
    assert np.isfinite(full).all();close(float(full.sum()),d['signed_sum'],1e-7)
    for field in ['net','positive','negative']:close(stats(full)[field],run['signed_summary'][field],1e-7)
    needle=needle_audit(w,info['keep'],case['gold'],run['signed_summary']['needle'])
    ids=np.asarray(r['input_freeze_before_model_load'][run['case']]['input_ids'],dtype=np.int64)
    eos=r['input_freeze_before_model_load'][run['case']]['target_ids'][-1]
    order=run['deletion_audit']['sorted_keep'];assert sorted(order)==sorted(info['keep']) and np.all(np.diff(w[order].astype(np.float64))<=0)
    groups=run['deletion_audit']['groups'];n,rem=divmod(len(order),20);cursor=0;deleted=set()
    assert len(groups)==20 and len(run['deletion_audit']['input_receipts'])==21
    for step,receipt in enumerate(run['deletion_audit']['input_receipts']):
        if step:
            group=order[cursor:cursor+n+(step<=rem)];cursor+=len(group)
            assert group==groups[step-1];deleted.update(group)
        changed=ids.copy();changed[sorted(deleted)]=eos
        assert receipt=={'input_sha256':hashlib.sha256(changed[None].tobytes()).hexdigest(),'deleted_positions':sorted(deleted)}
    memory=run['memory_cost']
    close(memory['peak_allocated_full_model_resident'],d['peak_allocated'])
    close(memory['peak_reserved_full_model_resident'],d['peak_reserved'])
    close(memory['peak_allocated_minus_before_pair'],d['peak_allocated']-run['GPU_allocated_before_pair'])
    close(memory['peak_allocated_minus_before_attribute'],d['peak_allocated']-run['GPU_allocated_before_attribute'])
    out['runs'].append({'case':run['case'],'method':run['method'],'phase':run['phase'],'number':run['number'],
        'vector_key':vk,'needle':needle,'signed_summary':stats(full),'eligible_signed':stats(w[info['keep']].astype(np.float64)),
        'memory_cost':memory,**{k:run[k] for k in ['GPU_allocated_before_pair','GPU_reserved_before_pair','GPU_allocated_before_attribute','GPU_reserved_before_attribute','GPU_allocated_after_cleanup','GPU_reserved_after_cleanup']},
        'outer_seconds':run['outer_attribute_seconds'],
        **{k:d[k] for k in ['complete_attribution_seconds_with_diagnostics','peak_allocated','peak_reserved','root_peak_allocated','root_effect','seed_effect','signed_sum','relative_residual']},
        'finite_callback_counts':run['finite_callback_counts'],'candidate_layer0_receipts':run['candidate_layer0_receipts']})
for key in p['quality_cases']:
    case=r['cases'][key]
    info=case['input'];keep=sorted(info['keep']);P=info['prompt_length'];T=info['total_length'];K=len(keep)
    summary={'input_sha256':info['input_sha256'],'methods':{},'fixed_control_masks':[],'historical_FT_endpoint_deltas_only':{}}
    for method,curve in case['curves'].items():
        run=next(x for x in r['runs'] if x['case']==key and x['method']==method)
        full=vectors[key+'_'+method+'_full'];w=vectors[key+'_'+method+'_evaluated'];order=curve['sorted_keep']
        assert full.shape==(T,) and w.shape==(P,) and np.array_equal(full[:P].astype(np.float32),w)
        close(float(full.sum()),run['details']['signed_sum'],1e-7)
        assert np.isfinite(full).all() and sorted(order)==keep and np.all(np.diff(w[order].astype(np.float64))<=0)
        assert order==run['deletion_audit']['sorted_keep'] and curve['input_receipts']==run['deletion_audit']['input_receipts']
        assert curve['status']=='complete' and curve['returned_forwards']==21 and len(curve['input_receipts'])==21
        n,extra=divmod(K,20);groups=[];cursor=0;selected=set()
        assert curve['input_receipts'][0]['deleted_positions']==[]
        for step in range(20):
            group=order[cursor:cursor+n+(step<extra)];cursor+=len(group);groups.append(group);selected.update(group)
            assert curve['input_receipts'][step+1]['deleted_positions']==sorted(selected)
        assert groups==run['deletion_audit']['groups'] and selected==set(keep)
        scores=np.asarray(curve['scores']);density=np.asarray(curve['density']);total=curve['attr_sum']
        exact_total=float(w[keep].astype(np.float64).sum());total_bound=gamma(K)*float(np.abs(w[keep].astype(np.float64)).sum())
        assert abs(total-exact_total)<=total_bound+1e-12
        theoretical_density=np.linspace(1,0,21) if total<=0 else np.r_[1,1-np.cumsum([w[g].astype(np.float64).sum()/total for g in groups])]
        density_bound=1e-12 if total<=0 else sum(gamma(len(g))*float(np.abs(w[g].astype(np.float64)).sum())/abs(total) for g in groups)+1e-12
        assert np.max(np.abs(density-theoretical_density))<=density_bound
        response,penalty,corrected,values=metric(scores,density)
        for name,value in [('normalized_model_response',response),('alignment_penalty',penalty),('corrected_scores',corrected),('return_metrics',values)]:close(value,curve[name])
        needle=None;gold=set(case['gold'])&set(keep)
        if gold:
            ww=np.maximum(w[keep],0);top=max(1,math.ceil(K*.1));threshold=np.sort(ww)[-top]
            above={keep[i] for i in np.where(ww>threshold)[0]};ties={keep[i] for i in np.where(ww==threshold)[0]};slots=top-len(above)
            low=(len(above&gold)+max(0,slots-len(ties-gold)))/len(gold);high=(len(above&gold)+min(slots,len(ties&gold)))/len(gold)
            assert low-1e-12<=curve['needle']<=high+1e-12;needle={'reported':curve['needle'],'independent_tie_aware_interval':[low,high],'gold_denominator':len(gold)}
        summary['methods'][method]={'RISE':values[0],'MAS':values[1],'alignment_augmented_AUC':values[2],'needle':needle,
            'scores':scores.tolist(),'density':density.tolist(),'normalized_model_response':response.tolist(),'alignment_penalty':penalty.tolist(),
            'attr_sum_reported_FP32':total,'attr_sum_exact_of_FP32_vector':exact_total,'density_recompute_max_delta':float(np.abs(density-theoretical_density).max()),'density_analytical_rounding_bound':density_bound,
            'eligible_signed':stats(w[keep].astype(np.float64)),'full_signed':stats(full),'response_signed':stats(full[P:]),
            'own_masks_pred_minus_actual':[float(w[x['deleted_positions']].astype(np.float64).sum()-(scores[0]-scores[i])) for i,x in enumerate(curve['input_receipts'])]}
    control=case['curves']['control'];candidate=case['curves']['candidate']
    for i in [0,20]:
        assert control['input_receipts'][i]==candidate['input_receipts'][i]
    summary['candidate_minus_control_native_endpoints']={str(i):candidate['scores'][i]-control['scores'][i] for i in [0,20]}
    assert case['clean_and_allEOS_scores_equal_between_methods']==[control['scores'][i]==candidate['scores'][i] for i in [0,20]]
    for entry in case['fixed_control_masks']['points']:
        step=entry['step'];mask=control['input_receipts'][step];assert entry['input_receipt']==mask
        actual=control['scores'][0]-control['scores'][step];close(actual,entry['actual_logprob_drop'])
        for method in ['control','candidate']:
            v=vectors[key+'_'+method+'_evaluated'][mask['deleted_positions']].astype(np.float64)
            close(v.sum(),entry[method]['deleted_signed_sum']);close(v.sum()-actual,entry[method]['prediction_minus_actual'])
        summary['fixed_control_masks'].append(entry)
    for label,historical in case.get('prior_fixed_FT_metrics',{}).items():
        if 'scores' in historical:
            summary['historical_FT_endpoint_deltas_only'][label]={'current_minus_historical_clean':control['scores'][0]-historical['scores'][0],
                'current_minus_historical_EOS':control['scores'][-1]-historical['scores'][-1],
                'scope':'Historical FT metrics are not recomputed or compared across different native endpoints.'}
    summary['candidate_minus_control']={name:summary['methods']['candidate'][name]-summary['methods']['control'][name] for name in ['RISE','MAS','alignment_augmented_AUC']}
    out['cases'][key]=summary
out['NI1_cost_only']={'warm_runs':r['warm_runs'],'methods':{},'vector_drift':[]}
for method in ['control','candidate']:
    rows=[x for x in r['runs'] if x['case']=='niah_mq_q2_1' and x['phase']=='measured' and x['method']==method]
    assert [x['number'] for x in rows]==([2,5] if method=='control' else [3,4])
    saved=r['cost_summary'][method];times=[x['outer_attribute_seconds'] for x in rows]
    assert saved['runs']==[x['number'] for x in rows] and saved['outer_seconds']==times
    close(saved['median_outer_seconds'],float(np.median(times)))
    assert saved['peak_allocated']==[x['details']['peak_allocated'] for x in rows]
    assert saved['peak_reserved']==[x['details']['peak_reserved'] for x in rows]
    out['NI1_cost_only']['methods'][method]={**saved,'mean_outer_seconds':float(np.mean(times)),
        'mean_runner_seconds':float(np.mean(saved['runner_seconds'])),'median_runner_seconds':float(np.median(saved['runner_seconds'])),
        'memory_costs':[x['memory_cost'] for x in rows]}
for row in r['runs']:
    if row['case']!='niah_mq_q2_1':continue
    method=row['method'];vk=row['vector_key'];oldkey='niah_mq_q2_1_'+method
    warm=next(x for x in r['runs'] if x['case']==row['case'] and x['method']==method and x['phase']=='warm')
    oldrun=next(x for x in prior['runs'] if x['case']==row['case'] and x['method']==method)
    oldcurve=prior['cases'][row['case']]['curves'][method]
    out['NI1_cost_only']['vector_drift'].append({'number':row['number'],'method':method,'phase':row['phase'],
        'full_vs_previous_wholepilot':drift(vectors[vk+'_full'],old_vectors[oldkey+'_full']),
        'evaluated_vs_previous_wholepilot':drift(vectors[vk+'_evaluated'],old_vectors[oldkey+'_evaluated']),
        'full_vs_same_run_warm':drift(vectors[vk+'_full'],vectors[warm['vector_key']+'_full']),
        'root_effect_minus_previous':row['details']['root_effect']-oldrun['details']['root_effect'],
        'old_wholepilot_same_deletion_input_count':sum(a==b for a,b in zip(row['deletion_audit']['input_receipts'],oldcurve['input_receipts']))})
cost=out['NI1_cost_only']['methods'];out['NI1_cost_only']['candidate_over_control']={
    name:cost['candidate'][name]/cost['control'][name] for name in ['mean_outer_seconds','median_outer_seconds','mean_runner_seconds','median_runner_seconds']}
out['NI1_cost_only']['previous_vectors_sha256']=sha(old_vectors_path)
for key,case in out['cases'].items():
    errors={m:np.asarray([x[m]['prediction_minus_actual'] for x in case['fixed_control_masks']]) for m in ['control','candidate']}
    case['fixed_control_error_summary']={'absolute_error_AUC':{m:auc(np.abs(v)) for m,v in errors.items()},
        'strictly_improved_interior_points':int((np.abs(errors['candidate'][1:20])<np.abs(errors['control'][1:20])).sum()),
        'interior_points':19}
out['decision']='This independent audit verifies the bounded pilot; metric deltas are candidate minus control and lower RISE/MAS is better. Promotion requires root review of both cases, signed fixed-mask errors and measured cost; no automatic followup or broader generalization claim.'
out['cost_scope']='Only NI1 measured C/S/S/C calls2,3,4,5 enter time or memory comparison, after separate C/S warmups0,1. Both include normal capture/checkpoints, endpoint copies, six means and the extra original FLA. Two observations per method provide a bounded fixed-input cost estimate, not a population or all-length claim.'
out['mask_hash_scope']='Own vectors independently reproduce every deletion set/order group; actual full-input hashes are verified against the pinned original scorer-hook receipts. Tokenizer/model execution is not repeated in this local analyzer.'
out['audit_seconds']=time.perf_counter()-started;target=A/'dt_layer0_fla_endpoint_average_stability_cost_summary_20260909.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'output':str(target),'sha256':sha(target),'seconds':out['audit_seconds'],'metric_deltas':{k:x['candidate_minus_control'] for k,x in out['cases'].items()}}))
