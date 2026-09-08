"""Read-only NumPy decomposition of actual MH0 curves; no new model inference."""
import hashlib,json,time
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';started=time.perf_counter()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_pristine_FT0_comparison_20260909_v1'
S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def auc(x):
    x=np.asarray(x,dtype=np.float64)
    return float((x.sum()-x[0]/2-x[-1]/2)/(len(x)-1))

def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return {'net':float(x.sum()),'positive':float(np.maximum(x,0).sum()),
        'negative':float(np.minimum(x,0).sum()),'absolute':float(np.abs(x).sum())}

def corrected(response,density):
    penalty=np.abs(response-density);raw=response+penalty;clipped=np.clip(raw,0,1)
    span=float(clipped.max()-clipped.min())
    final=(clipped-clipped.min())/span if span else np.linspace(1,0,len(raw))
    return {'response':response,'penalty':penalty,'uncorrected':raw,'corrected':final,
        'metrics':[auc(response),auc(final),auc(raw)]}

def original(scores,density):
    signed=(scores-scores[-1])/abs(scores[0]-scores[-1])
    clipped=np.clip(signed,0,1);response=np.minimum.accumulate(clipped)
    return dict(corrected(response,density),raw_signed_response=signed,clipped_response=clipped)

def density_on_receipts(w,total,receipts):
    # CPU64 contraction of saved FP32 weights on already executed masks. This
    # is an analytical crossing, not the author's own sorting or a new score.
    return np.asarray([1-float(w[item['deleted_positions']].sum())/total for item in receipts])

r=read(D/'results.json');p=read(D/'protocol.json');receipt=read(D/'terminal_receipt.json')
assert r['status']=='MH0_current_DT_pristine_FT0_1DT1Both42score_complete' and r['protocol']==p and receipt['pid_alive'] is False
for name,want in receipt['files'].items():assert sha(D/name)==want,name
for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['DT_entered']==r['DT_returned']==r['FT_entered']==r['FT_calls']==1
assert r['scorer_entered']==r['scorer_returned']==42 and r['generation_calls']==0
assert sha(D/'vectors.npz')==r['vectors_sha256']
official=R/'research/third_party/flashtrace_qwen35_e81b3be/flashtrace/improved.py'
raw=official.read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
assert blob==p['official_source_blob_sha1']['flashtrace/improved.py']
case=r['cases']['morehopqa_0'];keep=np.asarray(case['input']['keep']);vectors=np.load(D/'vectors.npz',allow_pickle=False)
out={'status':'MH0_observed_curve_and_signed_density_decomposition_verified',
    'results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),
    'vectors_sha256':sha(D/'vectors.npz'),'original_metric_source_blob_sha1':blob,
    'input':{k:v for k,v in case['input'].items() if k!='keep'},'methods':{},'crossed_existing_masks':{},
    'sign_convention':'Raw conditional error = deleted DT attribution minus actual clean-to-deleted logprob drop.',
    'analysis_scope':'Actual own original metrics plus exact scalar identities and CPU crossings of saved vectors/42executed masks. No cross entry is a new model counterfactual, reported official metric, score repair or candidate.'}
calculated={};weights={}
for method in ['FT0','DT']:
    curve=case['curves'][method];scores=np.asarray(curve['scores']);density=np.asarray(curve['density'])
    w=vectors['morehopqa_0_'+method+'_evaluated'].astype(np.float64);weights[method]=w
    assert w.shape==(case['input']['prompt_length'],) and np.isfinite(w).all()
    assert sorted(curve['sorted_keep'])==keep.tolist()
    assert np.all(np.diff(w[curve['sorted_keep']])<=0)
    total=float(curve['attr_sum']);assert total>0
    n,extra=divmod(len(keep),20);offset=0;deleted=set()
    for step in range(20):
        group=curve['sorted_keep'][offset:offset+n+(step<extra)];offset+=len(group);deleted.update(group)
        assert curve['input_receipts'][step+1]['deleted_positions']==sorted(deleted)
    rebuilt=density_on_receipts(w,total,curve['input_receipts'])
    assert np.max(np.abs(rebuilt-density))<2e-6
    value=original(scores,density);calculated[method]=value
    for key,field in [('response','normalized_model_response'),('penalty','alignment_penalty'),('corrected','corrected_scores'),('metrics','return_metrics')]:
        assert np.max(np.abs(np.asarray(value[key])-np.asarray(curve[field])))<1e-12,(method,field)
    response=value['response'];raw_response=value['raw_signed_response'];negative=np.maximum(-density,0)
    remaining=np.abs(response-np.maximum(density,0))
    assert np.max(np.abs(value['penalty']-negative-remaining))<1e-12
    raw_alignment=np.abs(raw_response-density)
    points=[];G=float(scores[0]-scores[-1])
    for step,item in enumerate(curve['input_receipts']):
        point={'step':step,'score':float(scores[step]),'raw_signed_response':float(raw_response[step]),
            'original_normalized_response':float(response[step]),'density':float(density[step]),
            'raw_signed_density_gap':float(density[step]-raw_response[step]),
            'original_alignment_penalty':float(value['penalty'][step]),
            'raw_signed_alignment':float(raw_alignment[step]),
            'response_processing_penalty_difference':float(value['penalty'][step]-raw_alignment[step]),
            'input_receipt':item}
        if method=='DT':
            predicted=float(w[item['deleted_positions']].sum());actual=float(scores[0]-scores[step]);error=predicted-actual
            point.update(deleted_signed_attribution=predicted,actual_raw_logprob_drop=actual,
                raw_conditional_error=error,negative_deleted_sum=float(np.minimum(w[item['deleted_positions']],0).sum()),
                density_gap_from_raw_error=-error/total,normalization_total_difference=actual*(1/G-1/total))
            assert abs(point['density_gap_from_raw_error']+point['normalization_total_difference']-(rebuilt[step]-raw_response[step]))<1e-12
        points.append(point)
    out['methods'][method]={'original_RISE':value['metrics'][0],'original_MAS':value['metrics'][1],
        'original_alignment_augmented_AUC':value['metrics'][2],'original_alignment_AUC':auc(value['penalty']),
        'negative_density_penalty_AUC':auc(negative),'remaining_alignment_AUC':auc(remaining),
        'corrected_postprocessing_AUC_change':value['metrics'][1]-value['metrics'][2],
        'raw_signed_alignment_AUC':auc(raw_alignment),
        'response_processing_alignment_AUC_change':auc(value['penalty'])-auc(raw_alignment),
        'raw_signed_response_AUC':auc(raw_response),'clipped_response_AUC':auc(value['clipped_response']),
        'original_monotonic_response_AUC':auc(response),'negative_raw_response_AUC':auc(np.maximum(-raw_response,0)),
        'raw_response_below_EOS_steps':[int(i) for i in np.where(raw_response<0)[0]],
        'negative_density_nonendpoint_steps':[int(i) for i in np.where(density[1:20]<0)[0]+1],
        'minimum_native_score':float(scores.min()),'minimum_raw_response':float(raw_response.min()),
        'minimum_density':float(density.min()),'eligible_signed_mass':stats(w[keep]),
        'reported_attr_sum':total,'exact_sum_of_saved_FP32_vector':float(w[keep].sum()),
        'density_recompute_max_difference':float(np.max(np.abs(rebuilt-density))),
        'scale_scope':'DT coefficients target logprob; FT0 targets original sink representation. Raw mass totals are not comparable units; normalized density crossing preserves each own denominator.',
        'points':points}
d=out['methods']['DT'];f=out['methods']['FT0']
parts={key:d[key]-f[key] for key in ['original_RISE','negative_density_penalty_AUC','remaining_alignment_AUC','corrected_postprocessing_AUC_change']}
gap=d['original_MAS']-f['original_MAS'];assert abs(sum(parts.values())-gap)<1e-12
out['exact_MAS_gap_decomposition']={'DT_minus_FT0_MAS':gap,'terms':parts,
    'identity':'|R-D| = max(-D,0)+|R-max(D,0)| because original R>=0. This partitions the original penalty; it does not recommend clipping signed coefficients or assume every negative density is wrong.'}
for vector_method in ['FT0','DT']:
    for trajectory in ['FT0','DT']:
        curve=case['curves'][trajectory]
        density=(np.asarray(curve['density']) if trajectory==vector_method else
            density_on_receipts(weights[vector_method],case['curves'][vector_method]['attr_sum'],curve['input_receipts']))
        value=original(np.asarray(curve['scores']),density)
        out['crossed_existing_masks'][vector_method+'_vector_on_'+trajectory+'_executed_masks']={
            'analytical_functional_AUC':value['metrics'][1],'observed_response_AUC':value['metrics'][0],
            'density':density.tolist(),'negative_density_AUC':auc(np.maximum(-density,0)),
            'scope':'Same actual scored deletion input for both density assignments; cross density is not sorted by its own vector, so this is not a newly measured official MAS.'}
cross=out['crossed_existing_masks'];M=lambda vector,order:cross[vector+'_vector_on_'+order+'_executed_masks']['analytical_functional_AUC']
allocation=.5*((M('DT','DT')-M('FT0','DT'))+(M('DT','FT0')-M('FT0','FT0')))
trajectory=.5*((M('DT','DT')-M('DT','FT0'))+(M('FT0','DT')-M('FT0','FT0')))
assert abs(allocation+trajectory-gap)<1e-12
out['symmetric_crossed_functional_accounting']={'allocation_component':allocation,'executed_mask_response_component':trajectory,
    'interaction':M('DT','DT')-M('FT0','DT')-M('DT','FT0')+M('FT0','FT0'),
    'limit':'Order and allocation interact; these are two-order arithmetic allocations of observed functional differences, not unique causal root causes or a model-intervention test.'}
sr=read(S/'results.json');sv=np.load(S/'vectors.npz',allow_pickle=False);assert sr['cases']['morehopqa_0']['input']==case['input']
old=sv['morehopqa_0_candidate_evaluated'].astype(np.float64);new=weights['DT']
out['prior_stability_source']={'results_sha256':sha(S/'results.json'),'vectors_sha256':sha(S/'vectors.npz'),
    'same_input':True,'vector_relative_L2':float(np.linalg.norm(new-old)/max(np.linalg.norm(old),1e-30)),
    'vector_max_absolute_difference':float(np.abs(new-old).max()),
    'old_control_MAS':sr['cases']['morehopqa_0']['curves']['control']['return_metrics'][1],
    'old_candidate_MAS':sr['cases']['morehopqa_0']['curves']['candidate']['return_metrics'][1]}
inventory=[]
for name in ['dt_conditional_boundaries_summary_20260908.json','dt_layer0_conditional_summary_20260908.json']:
    item=read(A/name);inventory.append({'artifact':name,'sha256':sha(A/name),'cases':list(item['cases']),
        'MH0_conditional_reuse':False})
old_decoder=A/'snapshot${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1/results.json';item=read(old_decoder)
assert item['input']['input_sha256']!=case['input']['input_sha256']
inventory.append({'artifact':str(old_decoder.relative_to(A)),'sha256':sha(old_decoder),'case_indices':item['protocol']['case_indices'],
    'input_sha256':item['input']['input_sha256'],'MH0_conditional_reuse':False})
out['layer_evidence_inventory']=inventory
out['current_MH0_B2_endpoint_layers']=r['runs'][0]['details']['layers']
out['verified_findings']=[
    'On this DT own deletion trajectory, signed density becomes negative at step3, while actual raw signed response remains positive through step13. True below-EOS responses are much shallower and do not explain the large negative density.',
    'Net attribution nearly matches the all-EOS logprob effect, but large positive and negative eligible masses cancel at the endpoint. This does not imply negative contributions are intrinsically invalid.',
    'RISE advantage applies to the predeclared trajectories only; step1 FT0 drops the actual score more than DT. Do not infer universally better ranking.',
    'Existing fine-grained layer ledgers concern MH1 or NI1. MH0 endpoint-only layers do not identify the conditional overprediction jump.']
out['next']={'study':'dt_MH0_current_conditional_boundaries_20260909.py','capture_source':'This standalone current DT own masks at steps0,3,10,20; never replace with the new diagnostic vector sorting.',
    'budget':{'model_loads':1,'original_NI0_eager_initializations':1,'current_DT_B2_attributions':1,
        'native_decoder_replays':32,'finite_FA':8,'public_FA_LSE':8,'finite_FLA_backend':25,'native_FLA_adjoint_stages':50,
        'original_B1_scorer_forwards':4,'FT':0,'generation':0,'new_candidates':0},
    'measurement':'Save actual coefficients on all34 normal-runner boundaries, capture actual clean/step3/step10/allEOS B1 boundaries, telescope actual-minus-predicted jumps across all32decoders and head/norm/input map. Report source-vector/native-score drift; select dominant actual layer without borrowing MH1/NI1 location.'}
out['actual_budget']={'new_GPU_calls':0,'new_model_calls':0,'new_FT_calls':0,'new_scorer_calls':0,
    'CPU_analysis_seconds':time.perf_counter()-started,'source_actual_job':{'DT':1,'FT_Both':1,'original_scorers':42,'wall_seconds':r['seconds']}}
out['analyzer_sha256']=sha(Path(__file__))
target=A/'dt_MH0_remaining_mismatch_summary_20260909.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'output':str(target),'sha256':sha(target),'MAS_gap':gap,'decomposition':parts,
    'DT_raw_below_EOS_steps':d['raw_response_below_EOS_steps'],'DT_min_raw_response':d['minimum_raw_response'],
    'DT_min_density':d['minimum_density'],'raw_negative_response_AUC':d['negative_raw_response_AUC'],
    'DT_raw_signed_alignment_AUC':d['raw_signed_alignment_AUC'],'DT_response_processing_alignment_change':d['response_processing_alignment_AUC_change'],
    'crossed_accounting':out['symmetric_crossed_functional_accounting'],'seconds':out['actual_budget']['CPU_analysis_seconds']}))
