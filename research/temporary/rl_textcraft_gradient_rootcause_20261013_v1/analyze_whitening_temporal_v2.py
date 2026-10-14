"""Intersect native rounding/physical bounds; analyze cached existing-data moments."""
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'infer_whitening_temporal_v1.json').read_bytes())['report']
scores=json.loads((HERE/'account_target_score_temporal_v1.json').read_bytes())['report']

def alpha(n,s,ss):return (1-s/n)/math.sqrt((ss-s*s/n)/(n-1)+1e-8)
def inverse(value,low,high,s,ss):
    for _ in range(80):
        mid=(low+high)/2
        if alpha(mid,s,ss)<value:low=mid
        else:high=mid
    return (low+high)/2

rows=[]
for old in raw['rows']:
    r={k:v for k,v in old.items() if k!='raw_files'}
    r['raw_sources_receipt']='infer_whitening_temporal_v1.json'
    if not r['truncated_successes']:
        low,high=r['success_policy_tokens'],r['full_history_policy_tokens']
        s,ss=r['success_raw_sum'],r['success_raw_sumsq']
        a=r['native_whitened_maximum_rounded']
        amin,amax=max(a-.0005,alpha(low,s,ss)),min(a+.0005,alpha(high,s,ss))
        if amin<=amax:
            lo=inverse(amin,low,high,s,ss);hi=inverse(amax,low,high,s,ss)
            n=(lo+hi)/2
            mean=s/n;std=math.sqrt((ss-s*s/n)/(n-1)+1e-8)
            r.update(inferred_count_rounding_interval=[lo,hi],inferred_retained_policy_count=n,
                inferred_raw_mean=mean,inferred_raw_std=std,
                inferred_zero_reward_whitened_coefficient=-mean/std,
                inferred_success_target_fraction=r['success_target_tokens']/n,
                common_global_token_mean_whitened_coefficient_L2=math.sqrt(n-1)/n,
                count_estimate_scope='Midpoint of intersection of printed maximum rounding and physical count bounds; not an exact count.')
            score=scores['temporal'][r['step']-1]
            r['DT_endpoint_common_token_mean_target_logit_gradient_L1_bounds']=[
                v*score['full_history_policy_tokens']/n for v in score['full_history_common_token_mean_target_logit_gradient_L1_bounds']]
    rows.append(r)

windows=[]
keys=['inferred_retained_policy_count','inferred_raw_mean','inferred_raw_std',
    'inferred_zero_reward_whitened_coefficient','inferred_success_target_fraction',
    'common_global_token_mean_whitened_coefficient_L2','native_grad_norm']
for lo,hi in [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,85)]:
    part=[r for r in rows[lo-1:hi] if r.get('inferred_retained_policy_count') is not None]
    windows.append({'iterations':[lo,hi],'inferred_iterations':len(part),
        'medians':{k:statistics.median(r[k] for r in part) for k in keys},
        'DT_endpoint_target_output_force_median_bounds':[
            statistics.median(r['DT_endpoint_common_token_mean_target_logit_gradient_L1_bounds'][i] for r in part) for i in (0,1)]})

report={'scope':__doc__,'model_GPU_calls':0,'rows':rows,'windows':windows,
    'usable_inferred_iterations':sum(r.get('inferred_retained_policy_count') is not None for r in rows),
    'excluded_truncated_success_iterations':[r['step'] for r in rows if r['truncated_successes']],
    'snapshot81_calibration':rows[80]['snapshot81_native_calibration'],
    'limits':raw['scope_limits']+[
        'Intervals are inferred from rounded values; they do not establish universal numerical exactness.',
        'Target output-force bounds use DT factual endpoint probabilities. Native HF endpoint calibration has nonzero differences (median absolute joint NLL difference about0.098 at iteration81); these are not strict native-actor historical bounds.',
        'Output-L1 and coefficient-L2 diagnostics have different norms and normalization from the native mean of preclip parameter norms. Their ratios cannot allocate causal percentages.',
        'Source/target direction, B4 placement and parameter Jacobian effects remain separate unknowns, to be localized through the already launched bounded native PG-group diagnostic.',
    ],
    'findings':[
        'Native raw-credit variance increases across the growth phase; a collapsing whitening denominator is not supported by these records.',
        'Retained policy-token volume decreases substantially. This changes numerical coefficient scales, but a scale change alone is not a measured explanation for parameter-gradient acceleration.',
        'Positive target fraction and target uncertainty also change. Their output-score effects can be bounded and calibrated from saved records rather than guessed from target fraction alone.',
    ],'rootcause_resolved':False,'accepted_repair':None,'deployed_repair':None}
path=HERE/'whitening-temporal-analysis-v2.json'
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'usable_inferred_iterations':report['usable_inferred_iterations'],'excluded':report['excluded_truncated_success_iterations'],'windows':windows,'calibration81':report['snapshot81_calibration']}))
