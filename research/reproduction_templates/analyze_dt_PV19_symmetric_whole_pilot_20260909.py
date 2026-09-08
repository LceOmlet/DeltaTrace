"""Independent CPU audit of the frozen symmetric layer19 PV pilot; no torch/model calls."""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from analyze_dt_original_regression_20260909 import close,stats,auc,needle,masks,curve_audit

A=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda b:hashlib.sha256(b).hexdigest()
local=lambda p:A/'snapshot'/str(p).lstrip('/')

def drift(new,old):
    new=np.asarray(new,dtype=np.float64);old=np.asarray(old,dtype=np.float64)
    assert new.shape==old.shape and np.isfinite(new).all() and np.isfinite(old).all()
    d=new-old;n=float(np.linalg.norm(old))
    return {'equal':bool(np.array_equal(new,old)),'relative_L2':float(np.linalg.norm(d))/n if n else None,
        'max_absolute':float(np.max(np.abs(d),initial=0)),'signed_sum_difference':float(d.sum())}

def provenance(D,p,r,receipt,complete):
    segment=p.get('segment_index')
    frozen='dt_PV19_symmetric_whole_pilot_protocol_20260909.json'
    assert p==r['protocol']==read(A/frozen)
    for key in ('pid_alive','proc_exists'):
        if key in receipt:assert receipt[key] is False
    for name,row in receipt['files'].items():
        assert sha(D/name)==(row if isinstance(row,str) else row['sha256']),name
        if isinstance(row,dict) and 'bytes' in row:assert (D/name).stat().st_size==row['bytes']
    for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
    with zipfile.ZipFile(D/'review_bundle.zip') as z:
        for name in z.namelist():assert z.read(name)==(D/name).read_bytes(),name
    assert not any('supported_secant' in name for name in p['files_sha256'])
    if complete:
        assert r['sources_before']==r['sources_after']
        assert r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
    source=r.get('sources_before',{});verified=[];local_snapshot_differences=[]
    for name,want in p['official_source_blob_sha1'].items():
        raw=(local(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want
        if source:assert source['official/'+name]==digest(raw)
        verified.append('official/'+name)
    for name,want in p['files_sha256'].items():
        if source:assert source[name]==want
    for name,want in p['runtime_source_sha256'].items():
        if source:assert source['native/'+name]==want
        f=local(p['isolated_site'])/name
        if f.exists():
            actual=sha(f)
            if actual==want:verified.append('native/'+name)
            else:
                fresh=A/'snapshot${ARTIFACT_ROOT}/codex_dt_current_FLA_source_receipts_20260909_v1';fr=read(fresh/'receipt.json');ff=fresh/Path(name).name
                assert fr[name]['original_path']==str(Path(p['isolated_site'])/name).replace('\\','/')
                assert fr[name]['sha256']==sha(ff)==want and ff.stat().st_size==fr[name]['bytes']
                verified.append(str(ff))
                local_snapshot_differences.append({'file':str(f),'local_sha256':actual,'executed_frozen_sha256':want,
                    'replacement_local_path':str(ff),'replacement_receipt_sha256':sha(fresh/'receipt.json'),
                    'scope':'Old snapshot preserved. Fresh read-only copy from the actual native environment independently matches the executed frozen hash; no unresolved source mismatch.'})
    for item in p['protected_sources']:
        if source:assert source[item['path']]==item['sha256']
        f=local(item['path'])
        if f.exists():assert sha(f)==item['sha256'];verified.append(item['path'])
    for name,want in p['span_source_sha256'].items():
        f=local(p['author_data_root'])/name
        assert digest(f.read_bytes().replace(b'\r\n',b'\n'))==want
        if source:assert source['author_spans/'+name]==want
    return {'independent_local_source_checks':verified,'local_historical_snapshot_differences':local_snapshot_differences,'native_model_sha256':p['native_model_sha256'],
        'native_FA_interface_sha256':p['installed_FA_interface_sha256'],
        'finite_FA_library_sha256':p['finite_FA_library_sha256'],
        'original_metric_blob_sha1':p['official_source_blob_sha1']['flashtrace/improved.py'],
        'sources_before_after_equal':r.get('sources_before')==r.get('sources_after') if complete else None,
        'checkpoint_scope':'Frozen config/tokenizer hashes and actual safetensors size/mtime guards in the executed source; full weight contents are not rehashed by this CPU audit.'}

def inputs_audit(p,r):
    inputs={};report={}
    for dataset,index in p['case_indices']:
        key=f'{dataset}_{index}';f=r['input_freeze_before_model_load'][key];info=f['input'];spec=p['fixed_records'][key]
        raw=local(p['cache_paths'][dataset]).read_bytes();assert digest(raw)==p['cache_hashes'][dataset]
        line=raw.decode().splitlines()[index];rec=json.loads(line);mapping=f['mapping']
        assert digest(line.encode())==spec['source_record_sha256']==mapping['source_record_sha256']
        assert digest(rec['prompt'].encode())==spec['prompt_text_sha256']
        assert digest(rec['target'].encode())==spec['fixed_target_text_sha256']
        assert mapping['original_cached_spans_reproduced'] and mapping['original_sink_span']==rec['sink_span']
        assert mapping['gold_function']=='unchanged author ruler_gold_prompt_token_indices'
        if spec.get('expected_input') is not None:assert info==spec['expected_input']
        if spec.get('expected_gold') is not None:assert mapping['gold']==spec['expected_gold']
        ids=np.asarray(f['input_ids'],dtype=np.int64);target=np.asarray(f['target_ids'],dtype=np.int64)
        assert ids.shape==(info['total_length'],) and target.shape==(info['target_length'],)
        assert np.array_equal(ids[info['prompt_length']:],target) and digest(ids.tobytes())==info['input_sha256']
        base=ids.copy();base[info['keep']]=target[-1]
        assert digest(base.tobytes())==f['baseline_sha256'] and np.flatnonzero(base!=ids).tolist()==info['keep']
        if key in r['cases']:
            case=r['cases'][key];assert case['input']==info and case['mapping']==mapping and case['gold']==mapping['gold']
            assert case['baseline_sha256']==f['baseline_sha256']
        inputs[key]=(ids,int(target[-1]))
        report[key]={'input':info,'record_sha256':spec['source_record_sha256'],'fixed_target_text_sha256':spec['fixed_target_text_sha256'],
            'gold':mapping['gold'],'baseline_sha256':f['baseline_sha256'],
            'scope':'Original author cache IDs, unchanged target suffix, author span remap receipt and eligible-only EOS replacement checked. No tokenizer/model reexecution.'}
    return inputs,report

def run_audit(run,p,r,z,inputs):
    assert run['status']=='complete' and run['root_forwards']==1
    d=run['details'];counts=run['counts'];method=run['method']
    assert {k:counts[k] for k in ('native_root','native_decoder_replays','finite_decoder_calls','public_FA_auxiliary_calls','finite_FA_calls','finite_FLA_calls')}=={
        'native_root':1,'native_decoder_replays':32,'finite_decoder_calls':32,'public_FA_auxiliary_calls':8,'finite_FA_calls':8,'finite_FLA_calls':24}
    assert counts['finite_FLA_backend_calls']==25
    assert counts['finite_FA_backend_calls']==8+(method=='candidate')
    assert run['finite_callback_counts']=={'FA_entered':8+(method=='candidate'),'FA_returned':8+(method=='candidate'),'FLA_backend_entered':25,'FLA_backend_returned':25}
    assert run['native_stage_accounting']=={'returned_FLA_backend_calls_times_two':50,'stages_inside_nonreturned_backend':0}
    assert d['norm_gate_rules']=={'0':'symmetric'} and d['finite_fla_by_layer']==[0] and d['select_output_rows'] is True
    assert d['attention_pv_rules']==run['PV_rule_scope']==({'19':'symmetric'} if method=='candidate' else {})
    assert run['finite_FA_phase_accounting']=={'callback_sites':8,'returned_calls':8+(method=='candidate'),'phases_from_returned_calls':3*(8+(method=='candidate')),'library':p['finite_FA_library']}
    assert len(run['candidate_layer0_receipts'])==1
    wrap=run['candidate_layer0_receipts'][0];assert wrap['status']=='returned' and wrap['endpoint_permutation']==[1,0]
    assert [c['orientation'] for c in wrap['backend_calls']]==['original','swapped']
    assert all(c['status']=='returned' for c in wrap['backend_calls'])
    assert wrap in r['finite_counts']['layer0_average_wrapper']['calls']
    kinds=[x['kind'] for x in d['calls']]
    for prefix in ('native_replay_','finite_decoder_'):
        assert [x for x in kinds if x.startswith(prefix)]==[prefix+str(i) for i in reversed(range(32))]
    assert [x for x in kinds if x.startswith('public_FA_LSE_')]==['public_FA_LSE_'+str(i) for i in reversed(p['expected_FA_layers'])]
    assert list(d['layers'])==[str(i) for i in reversed(range(32))]
    for i,layer in d['layers'].items():
        assert layer['decoder_calls']=={k:1 for k in ('input_norm','post_norm','gate','silu','up','down','mlp','decoder')}
        assert layer['mixer_calls']==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if layer['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
        assert (layer['block_type']=='full_attention')==(int(i) in p['expected_FA_layers'])
    key=run['case'];info=r['cases'][key]['input'];full=z[run['vector_key']+'_full'];w=z[run['vector_key']+'_evaluated']
    assert full.dtype==np.float64 and w.dtype==np.float32
    assert full.shape==(info['total_length'],) and w.shape==(info['prompt_length'],) and np.array_equal(full[:len(w)].astype(np.float32),w)
    close(full.sum(),d['signed_sum'],1e-7)
    for field in ('net','positive','negative'):close(stats(full)[field],run['signed_summary'][field],1e-7)
    nt=needle(w,info['keep'],r['cases'][key]['gold'],run['signed_summary']['needle'])
    masks(w,info,*inputs[key],run['deletion_audit'])
    lp0=np.asarray(d['target_logp0']);lp1=np.asarray(d['target_logp1'])
    assert lp0.shape==lp1.shape==(info['target_length'],)
    assert d['selected_predictor_rows']==list(range(info['prompt_length']-1,info['total_length']-1))
    close((lp1-lp0).sum(),d['root_effect'],1e-7)
    close(d['compiled_seed_logprob_effect']-d['root_effect'],d['compiled_seed_logprob_effect_minus_root'],1e-7)
    mem=run['memory_cost'];assert mem['peak_allocated_full_model_resident']==d['peak_allocated']
    assert mem['peak_reserved_full_model_resident']==d['peak_reserved']
    assert mem['peak_allocated_minus_before_pair']==d['peak_allocated']-run['GPU_allocated_before_pair']
    assert mem['peak_allocated_minus_before_attribute']==d['peak_allocated']-run['GPU_allocated_before_attribute']
    return {'case':key,'method':method,'rule':d['attention_pv_rules'],'needle':nt,'outer_seconds':run['outer_attribute_seconds'],
        'runner_seconds':d['complete_attribution_seconds_with_diagnostics'],'memory_cost':mem,
        'GPU_allocated_after_cleanup':run['GPU_allocated_after_cleanup'],'GPU_reserved_after_cleanup':run['GPU_reserved_after_cleanup'],
        **{k:d[k] for k in ('root_effect','seed_effect','signed_sum','relative_residual','compiled_seed_logprob_effect','compiled_seed_logprob_effect_minus_root')},
        'finite_callback_counts':run['finite_callback_counts'],'layer0_average_receipt':wrap}

def audit(D):
    tick=time.perf_counter();D=Path(D);p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
    segment=p.get('segment_index');expected_status=('layer19_PV_symmetric_remaining_segment_4DT100FLA84score_complete' if segment is not None else 'layer19_PV_content0_current_layer0_4DT100FLA84score_complete')
    complete=r['status']==expected_status
    if segment is not None:
        pairs=[['niah_mq_q2_0','morehopqa_2']]
        assert type(segment) is int and segment==0 and p['quality_cases']==pairs[segment]
        first,second=pairs[segment]
        assert p['call_schedule']==[[first,'control','quality'],[first,'candidate','quality'],[second,'candidate','quality'],[second,'control','quality']]
    out={'status':'independent_PV19_symmetric_whole_pilot_audit_passed' if complete else 'independent_PV19_whole_pilot_partial_failure_audit',
        'study_status':r['status'],'segment_index':segment,'error':r.get('error'),'analyzer_sha256':sha(__file__),
        'metric_helper_sha256':sha(A/'analyze_dt_original_regression_20260909.py'),
        'results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'receipt_sha256':sha(D/'terminal_receipt.json'),
        'wall_seconds':r['seconds'],'frozen_budget':p['budget'],'sources':provenance(D,p,r,receipt,complete),
        'actual_counters':{k:r[k] for k in ('model_loads','native_eager_diagnostics','DT_entered','DT_returned','scorer_entered','scorer_returned','FT_calls','generation_calls')},
        'finite_counters':r['finite_counts'],'cases':{},'runs':[]}
    for prefix,maximum in [('DT',4),('scorer',84)]:assert 0<=r[prefix+'_returned']<=r[prefix+'_entered']<=maximum
    if not complete:
        out['partial_runs']=r['runs'];out['scope']='Partial actual counters only; nonreturned operator internals are unknown. No candidate quality conclusion.'
        return out
    assert r['model_loads']==r['native_eager_diagnostics']==1 and r['FT_calls']==r['generation_calls']==0
    assert r['DT_entered']==r['DT_returned']==4 and r['scorer_entered']==r['scorer_returned']==84
    assert [[x['case'],x['method'],x['phase']] for x in r['runs']]==p['call_schedule']
    for method in ('control','candidate'):assert r['finite_counts'][method]=={'entered':16 if method=='control' else 18,'returned':16 if method=='control' else 18}
    assert r['finite_counts']['FLA_backend']=={'entered':100,'returned':100,'native_adjoint_stages_from_returned_calls':200,'native_stages_inside_nonreturned_calls':0}
    wrapped=r['finite_counts']['layer0_average_wrapper'];assert wrapped['entered']==wrapped['returned']==len(wrapped['calls'])==4
    assert r['finite_counts']['original_FA_phases_from_returned_calls']==102
    assert sha(D/'vectors.npz')==r['vectors_sha256'];out['vectors_sha256']=r['vectors_sha256']
    z=np.load(D/'vectors.npz',allow_pickle=False);assert set(z.files)=={k+'_'+m+'_'+part for k in p['quality_cases'] for m in ('control','candidate') for part in ('full','evaluated')}
    inputs,out['input_provenance']=inputs_audit(p,r)
    out['runs']=[run_audit(x,p,r,z,inputs) for x in r['runs']]
    out['counts']={'models':1,'eager_NI0_initializations':1,'DT':4,'native_replays':128,'finite_decoders':128,'finite_FA':34,
        'public_FA_LSE':32,'finite_FA_phases':102,'content0_calls':2,'content1_calls':32,'finite_FLA':100,'native_FLA_adjoint_stages':200,
        'common_layer0_average_calls':4,'original_score_forwards':84,'FT':0,'generation':0}
    assert set(r['cases'])==set(p['quality_cases'])
    for key,case in r['cases'].items():
        methods={};rr={m:next(x for x in r['runs'] if x['case']==key and x['method']==m) for m in ('control','candidate')}
        for m in rr:
            # Alias only for the pure CPU helper's DT-units branch; vectors unchanged.
            alias='DT_'+m;vv={key+'_'+alias+'_'+part:z[key+'_'+m+'_'+part] for part in ('full','evaluated')}
            methods[m]=curve_audit(key,alias,case['curves'][m],case,rr[m]['deletion_audit'],vv,inputs)
        ctrl=case['curves']['control'];cand=case['curves']['candidate'];same=[]
        assert len(case['fixed_control_masks']['points'])==21
        for step,entry in enumerate(case['fixed_control_masks']['points']):
            receipt=ctrl['input_receipts'][step];assert entry['step']==step and entry['input_receipt']==receipt
            actual=ctrl['scores'][0]-ctrl['scores'][step];close(entry['actual_logprob_drop'],actual)
            row={'step':step,'input_receipt':receipt,'actual_logprob_drop':actual}
            for m in rr:
                v=z[key+'_'+m+'_evaluated'][receipt['deleted_positions']].astype(np.float64)
                close(v.sum(),entry[m]['deleted_signed_sum']);close(v.sum()-actual,entry[m]['prediction_minus_actual'])
                row[m]={**entry[m],'deleted_signed_parts':stats(v)}
            row['candidate_minus_control_error']=row['candidate']['prediction_minus_actual']-row['control']['prediction_minus_actual']
            same.append(row)
        for step in (0,20):assert ctrl['input_receipts'][step]==cand['input_receipts'][step]
        endpoint=[cand['scores'][i]-ctrl['scores'][i] for i in (0,20)]
        assert case['clean_and_allEOS_scores_equal_between_methods']==[v==0 for v in endpoint]
        errors={m:np.asarray([x[m]['prediction_minus_actual'] for x in same]) for m in rr}
        delta={name:methods['candidate'][name]-methods['control'][name] for name in ('RISE','MAS','alignment_augmented_AUC')}
        delta['needle']=methods['candidate']['needle']['reported']-methods['control']['needle']['reported'] if methods['control']['needle'] else None
        mas_parts=('RISE','negative_density_AP_extra_AUC','AP_with_nonnegative_density_AUC','original_postprocessing_change_AUC')
        mas_delta={k:methods['candidate']['MAS_decomposition'][k]-methods['control']['MAS_decomposition'][k] for k in mas_parts}
        close(sum(mas_delta.values()),delta['MAS'])
        d0=rr['control']['details'];d1=rr['candidate']['details'];full0=z[key+'_control_full'];full1=z[key+'_candidate_full']
        keysummary={'methods':methods,'candidate_minus_control':delta,'candidate_minus_control_MAS_decomposition':mas_delta,'fixed_control_masks':same,
            'same_control_error_summary':{m:{'MAE_AUC':auc(np.abs(e)),'mean_absolute_all21':float(np.mean(np.abs(e))),
                'max_absolute_all21':float(np.max(np.abs(e))),'signed_error_AUC':auc(e),'points':{str(i):float(e[i]) for i in (1,3,10,20)}} for m,e in errors.items()},
            'interior_absolute_error_improved_steps':(np.flatnonzero(np.abs(errors['candidate'][1:20])<np.abs(errors['control'][1:20]))+1).tolist(),
            'interior_absolute_error_worse_steps':(np.flatnonzero(np.abs(errors['candidate'][1:20])>np.abs(errors['control'][1:20]))+1).tolist(),
            'same_scored_input_steps_between_own_curves':[i for i in range(21) if ctrl['input_receipts'][i]==cand['input_receipts'][i]],
            'candidate_minus_control_native_score_endpoints':endpoint,
            'same_case_native_root_drift':{k:drift(d1[k],d0[k]) for k in ('target_logp0','target_logp1')},
            'same_case_root_seed_deltas':{k:d1[k]-d0[k] for k in ('root_effect','seed_effect','compiled_seed_logprob_effect')},
            'candidate_vector_change_not_pure_numerical_drift':{'full':drift(full1,full0),'evaluated':drift(z[key+'_candidate_evaluated'],z[key+'_control_evaluated']),
                'sign_changed_coordinates':int(np.count_nonzero(np.sign(full1)!=np.sign(full0))),'change_signed_parts':stats(full1-full0)},
            'regressions':[k for k,v in delta.items() if v is not None and ((k=='needle' and v<0) or (k!='needle' and v>0))]}
        if segment is not None:
            ref=p['historical_FT_references'][key];assert ref==case['historical_FT_reference']
            assert ref['input_sha256']==case['input']['input_sha256'] and ref['gold']==case['gold']
            assert sha(local(ref['results_path']))==ref['results_sha256'] and sha(local(ref['protocol_path']))==ref['protocol_sha256']
            old=read(local(ref['results_path']))['cases'][key];assert old['input']==case['input'] and old['gold']==case['gold']
            for m in ('FT0','FT3'):
                assert old['curves'][m]['return_metrics']==ref['curves'][m]['original_return_metrics']
                assert old['curves'][m]['needle']==ref['curves'][m]['needle']
            keysummary['historical_FT_reference_only']=ref
            keysummary['current_control_minus_historical_FT_native_endpoints']={m:[ctrl['scores'][i]-old['curves'][m]['scores'][i] for i in (0,20)] for m in ('FT0','FT3')}
        cross={}
        for background in ('control','candidate'):
            cr=case['curves'][background];err={}
            for method in ('control','candidate'):
                value=z[key+'_'+method+'_evaluated'].astype(np.float64)
                error=np.asarray([value[rec['deleted_positions']].sum()-(cr['scores'][0]-cr['scores'][step]) for step,rec in enumerate(cr['input_receipts'])])
                err[method]={'MAE_AUC':auc(np.abs(error)),'signed_error_AUC':auc(error),'errors':error.tolist()}
            cross[background]=err
        for method in ('control','candidate'):close(cross['control'][method]['MAE_AUC'],keysummary['same_control_error_summary'][method]['MAE_AUC'])
        keysummary['both_actual_mask_families']=cross
        out['cases'][key]=keysummary
    out['metric_scope']='Unchanged author RISE/MAS, lower is better; exact original formula recomputation and analytical FP32 density bounds, with all negative scores retained. Own-curve changes are not same-condition error repairs. The separate fixed-control-mask ledger uses one actual scored control input per point for both vectors.'
    out['decision_scope']='This execution covers only its two previously used development cases, not by itself full eight-case acceptance or generalization. Report every regression and residual; no automatic promotion or further GPU call.'
    out['cost_scope']='One call per method/shape in opposite orders; includes native root/replays, endpoint copies, existing finite operators and diagnostics. Shape compilation/order confounds remain; not a warmed speed or true minibatch measurement.'
    out['audit_seconds']=time.perf_counter()-tick
    return out

if __name__=='__main__':
    result=audit(A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV19_symmetric_whole_pilot_20260909_v1')
    target=A/'dt_PV19_symmetric_whole_pilot_summary_20260909.json';target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'output':str(target),'sha256':sha(target),'status':result['status'],'deltas':{k:v['candidate_minus_control'] for k,v in result['cases'].items()},'metrics':{k:{m:{x:q[x] for x in ['RISE','MAS','needle']} for m,q in v['methods'].items()} for k,v in result['cases'].items()}}))
