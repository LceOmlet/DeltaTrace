"""CPU-only original-metric audit of saved true-batch/singleton NI vectors."""
import argparse,json,time
from pathlib import Path
import numpy as np
from analyze_dt_original_regression_20260909 import close,stats,needle,masks,original_metric,gamma,auc
from analyze_dt_true_minibatch_NI1_46_20260909 import source_audit,input_audit,sha,digest,local,read

A=Path(__file__).resolve().parent
SUCCESS='true_minibatch_saved_vectors_4original_curves84scores_complete'

def audit(D):
    tick=time.perf_counter();D=Path(D);p=read(D/'protocol.json');r=read(D/'results.json');success=r['status']==SUCCESS
    assert r['protocol']==p==read(A/'dt_true_minibatch_original_metrics_protocol_20260909.json')
    sr=p['source_batch'];source={}
    for name in ('results','vectors','protocol'):
        path=local(sr[name+'_path']);assert sha(path)==sr[name+'_sha256'];source[name]=path
    old=read(source['results']);oldp=read(source['protocol']);assert old['protocol']==oldp
    assert old['status']=='true_distinct_NI1_46_batch2_9DT225FLA_no_scores_complete'
    for key in p['same_native_identity_keys']:assert p[key]==oldp[key]
    provenance=input_audit(oldp,old)
    out={'status':'independent_true_minibatch_original_metric_audit_passed' if success else 'independent_failed_true_minibatch_metric_partial_audit',
        'study_status':r['status'],'study_error':r.get('error'),'analyzer_sha256':sha(__file__),
        'helper_sha256':{name:sha(A/name) for name in ('analyze_dt_original_regression_20260909.py','analyze_dt_true_minibatch_NI1_46_20260909.py')},
        'protocol_sha256':sha(D/'protocol.json'),'results_sha256':sha(D/'results.json'),'sources':source_audit(D,p,r),
        'source_batch':sr,'input_provenance':provenance,'actual_seconds':r['seconds'],'frozen_budget':p['budget'],'cases':{}}
    if 'input_freeze_before_model_load' in r:assert r['input_freeze_before_model_load']==old['input_freeze_before_model_load']
    for key in ('DT_entered','DT_returned','finite_FA_calls','finite_FLA_calls','FT_calls','generation_calls'):assert r[key]==0
    assert 0<=r['scorer_returned']<=r['scorer_entered']<=84
    assert r['model_loads']<=1 and r['native_eager_diagnostics']<=1
    out['actual_counts']={k:r[k] for k in ('model_loads','native_eager_diagnostics','DT_entered','DT_returned','finite_FA_calls','finite_FLA_calls','FT_calls','generation_calls','scorer_entered','scorer_returned')}
    if success:assert r['model_loads']==r['native_eager_diagnostics']==1 and r['scorer_entered']==r['scorer_returned']==84
    original=np.load(source['vectors'],allow_pickle=False)
    z=np.load(D/'vectors.npz',allow_pickle=False) if (D/'vectors.npz').exists() else None
    out['vectors_sha256']=sha(D/'vectors.npz') if z is not None else None
    if 'vectors_sha256' in r:assert out['vectors_sha256']==r['vectors_sha256']
    total_entered=total_returned=0;completed=[]
    for key,method in p['quality_schedule']:
        if key not in r['cases'] or method not in r['cases'][key]['curves']:continue
        case=r['cases'][key];f=old['input_freeze_before_model_load'][key];info=f['input'];keep=info['keep'];K=len(keep)
        assert case['input']==info and case['mapping']==f['mapping'] and case['gold']==f['mapping']['gold']
        curve=case['curves'][method];entered=curve['entered_forwards'];returned=curve['returned_forwards']
        assert 0<=returned<=entered<=21;total_entered+=entered;total_returned+=returned
        row=out['cases'].setdefault(key,{'methods':{},'actual_endpoint_scores':{}})
        item={'status':curve['status'],'scorer_entered':entered,'scorer_returned':returned};row['methods'][method]=item
        number=p['saved_warm_runs'][key][method];run=old['runs'][number]
        assert run['status']=='complete' and run['phase']=='warm' and run['mode']==('batch2' if method=='batch2' else 'single_pair')
        record=next(s for s in run['sample_records'] if s['case']==key);name=key+'_'+method
        expected=record['deletion_audit'];full=z[name+'_full'];w=z[name+'_evaluated']
        assert full.dtype==np.float64 and w.dtype==np.float32 and full.shape==(1201,) and w.shape==(937,)
        assert np.array_equal(full,original[record['vector_key']+'_full']) and np.array_equal(w,original[record['vector_key']+'_evaluated'])
        assert np.array_equal(full[:937].astype(np.float32),w) and np.isfinite(full).all()
        for part,array in [('full',full),('evaluated',w)]:assert digest(array.tobytes())==p['saved_vector_sha256'][key][method][part]
        src=r['saved_vector_provenance'][name];assert src['source_run']==number and src['source_vector_key']==record['vector_key']
        assert src['deletion_audit']==expected and src['source_example_batch_size']==run['example_batch_size']
        for part in ('full','evaluated'):assert src[part+'_sha256']==p['saved_vector_sha256'][key][method][part]
        ids=np.asarray(f['input_ids'],dtype=np.int64);masks(w,info,ids,f['target_ids'][-1],expected)
        assert curve['input_receipts']==expected['input_receipts'][:len(curve['input_receipts'])]
        assert curve['evaluated_vector_sha256']==p['saved_vector_sha256'][key][method]['evaluated']
        recovered=needle(w,keep,case['gold'],curve['needle']);close(curve['needle'],record['needle'])
        item.update(needle=recovered,full_signed=stats(full),eligible_signed=stats(w[keep]),source_run=number)
        if curve['status']!='complete':assert not success;continue
        assert entered==returned==21 and curve['sorted_keep']==expected['sorted_keep']
        assert curve['native_logits_dtype']=='torch.bfloat16' and curve['native_logits_shape'][:2]==[1,1201]
        scores=np.asarray(curve['scores'],dtype=np.float64);density=np.asarray(curve['density'],dtype=np.float64)
        assert scores.shape==density.shape==(21,) and np.isfinite(scores).all() and np.isfinite(density).all()
        total=curve['attr_sum'];exact=float(w[keep].astype(np.float64).sum());bound=gamma(K)*float(np.abs(w[keep].astype(np.float64)).sum())
        assert abs(total-exact)<=bound+1e-12
        groups=expected['groups'];pred=np.linspace(1,0,21) if total<=0 else np.r_[1,1-np.cumsum([w[g].astype(np.float64).sum()/total for g in groups])]
        density_bound=1e-12 if total<=0 else sum(gamma(len(g))*float(np.abs(w[g].astype(np.float64)).sum())/abs(total) for g in groups)+1e-12
        maxdelta=float(np.max(np.abs(pred-density)));assert maxdelta<=density_bound
        response,penalty,corrected,values,fallback=original_metric(scores,density)
        for field,value in [('normalized_model_response',response),('alignment_penalty',penalty),('corrected_scores',corrected),('return_metrics',values),('observed_original_return_metrics',values)]:close(curve[field],value)
        errors=[float(w[x['deleted_positions']].astype(np.float64).sum()-(scores[0]-scores[i])) for i,x in enumerate(expected['input_receipts'])]
        item.update(RISE=values[0],MAS=values[1],alignment_augmented_AUC=values[2],scores=scores.tolist(),density=density.tolist(),
            normalized_model_response=response.tolist(),alignment_penalty=penalty.tolist(),corrected_scores=corrected.tolist(),
            constant_corrected_curve_fallback=fallback,attr_sum_reported_FP32=total,attr_sum_exact_of_FP32_vector=exact,
            density_recompute_max_delta=maxdelta,density_analytical_rounding_bound=density_bound,
            own_masks_prediction_minus_actual=errors,input_receipts=curve['input_receipts'])
        row['actual_endpoint_scores'][method]=[float(scores[0]),float(scores[-1])];completed.append([key,method])
    assert total_entered==r['scorer_entered'] and total_returned==r['scorer_returned']
    if success:
        assert completed==p['quality_schedule'] and len(r['calls'])==6
        assert set(z.files)=={key+'_'+method+'_'+part for key,method in p['quality_schedule'] for part in ('full','evaluated')}
    for key,row in out['cases'].items():
        if not all(row['methods'].get(m,{}).get('status')=='complete' for m in ('singleton','batch2')):continue
        one=row['methods']['singleton'];batch=row['methods']['batch2']
        for i in (0,20):assert one['input_receipts'][i]==batch['input_receipts'][i]
        row['batch_minus_singleton']={name:batch[name]-one[name] for name in ('RISE','MAS','alignment_augmented_AUC')}
        row['batch_minus_singleton']['needle']=batch['needle']['reported']-one['needle']['reported']
        row['batch_curve_minus_singleton_curve_native_endpoint_scores']=[b-a for a,b in zip(row['actual_endpoint_scores']['singleton'],row['actual_endpoint_scores']['batch2'])]
        saved=r['cases'][key];assert saved['actual_endpoint_scores']==row['actual_endpoint_scores']
        close(saved['endpoint_batch_curve_minus_singleton_curve'],row['batch_curve_minus_singleton_curve_native_endpoint_scores'])
        close(saved['batch_vector_minus_singleton_vector']['return_metrics'],[row['batch_minus_singleton'][x] for x in ('RISE','MAS','alignment_augmented_AUC')])
        close(saved['batch_vector_minus_singleton_vector']['needle'],row['batch_minus_singleton']['needle'])
        # Fixed singleton masks share their measured B1 effects between both
        # saved coefficient vectors; this is a CPU contraction, not a new score.
        row['same_singleton_masks']={}
        for method in ('singleton','batch2'):
            w=z[key+'_'+method+'_evaluated'];error=np.asarray([float(w[x['deleted_positions']].astype(np.float64).sum()-(one['scores'][0]-one['scores'][i])) for i,x in enumerate(one['input_receipts'])])
            row['same_singleton_masks'][method]={'prediction_minus_actual':error.tolist(),'absolute_error_AUC':auc(np.abs(error))}
    out['actual_counts']['complete_original_curves']=len(completed)
    out['scope']='Original four B1 scoring curves for immutable warm singleton and genuine two-example attribution vectors. RISE/MAS lower is better. Own signed density and actual deletion receipts independently verified, no negative clamping except unchanged author needle calculation. No attribution or native replay; no score cache. This audit does not validate arbitrary lengths, datasets or batched scoring.'
    out['study_cost_scope']=r.get('cost_scope');out['audit_seconds']=time.perf_counter()-tick
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',nargs='?',default=str(A/'snapshot${ARTIFACT_ROOT}/codex_dt_true_minibatch_original_metrics_20260909_v1'));args=ap.parse_args()
    result=audit(args.directory);target=A/'dt_true_minibatch_original_metrics_summary_20260909.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'output':str(target),'sha256':sha(target),'status':result['status'],
        'deltas':{k:v.get('batch_minus_singleton') for k,v in result['cases'].items()}},ensure_ascii=False))
