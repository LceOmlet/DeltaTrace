"""Analyze actual dump placement, count uncertainty, and native endpoint weights."""
import json
import statistics
from pathlib import Path
from datetime import datetime

HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'account_historical_B4_weights_v1.json').read_bytes())['report']
white=json.loads((HERE/'whitening-temporal-analysis-v2.json').read_bytes())
rows=[]
for r,w in zip(raw['rows'],white['rows']):
    assert r['step']==w['step']
    out={k:r[k] for k in ['step','native_grad_norm','truncated_successes','positive_rows','total_policy_count_approx']}
    if w.get('inferred_retained_policy_count'):
        out['inferred_count_interval']=w['inferred_count_rounding_interval']
        out['approx_count_relative_error_to_inferred_midpoint']=r['total_policy_count_approx']/w['inferred_retained_policy_count']-1
        out['inferred_whitened_target_alpha']=(1-w['inferred_raw_mean'])/w['inferred_raw_std']
    out['B4_policy_count_range']=[min(x['policy_count'] for x in r['B4']),max(x['policy_count'] for x in r['B4'])]
    for mode in ['legacy','common']:
        out[f'target_coefficient_mass_{mode}_mean']=statistics.mean(x[f'target_coefficient_mass_{mode}'] for x in r['global64'])
        out[f'target_logit_L1_{mode}_DT_endpoint_mean_bounds']=[statistics.mean(x[f'target_logit_L1_bounds_{mode}_DT_endpoint'][j] for x in r['global64']) for j in (0,1)]
    out['target_mass_B4_vs_common_ratio']=out['target_coefficient_mass_legacy_mean']/out['target_coefficient_mass_common_mean']
    out['target_logit_L1_B4_vs_common_lower_ratio']=out['target_logit_L1_legacy_DT_endpoint_mean_bounds'][0]/out['target_logit_L1_common_DT_endpoint_mean_bounds'][0]
    out['target_logit_L1_B4_vs_common_upper_ratio']=out['target_logit_L1_legacy_DT_endpoint_mean_bounds'][1]/out['target_logit_L1_common_DT_endpoint_mean_bounds'][1]
    out['suspect_large_decoded_B4s']=sum(x['suspect_large_decoded_turn'] for x in r['B4'])
    rows.append(out)
keys=['native_grad_norm','total_policy_count_approx','inferred_whitened_target_alpha',
    'target_coefficient_mass_legacy_mean','target_coefficient_mass_common_mean','target_mass_B4_vs_common_ratio',
    'target_logit_L1_B4_vs_common_lower_ratio','target_logit_L1_B4_vs_common_upper_ratio']
windows=[]
for lo,hi in [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,85)]:
    part=[r for r in rows[lo-1:hi] if not r['truncated_successes']]
    windows.append({'iterations':[lo,hi],'n':len(part),'medians':{k:statistics.median(r[k] for r in part) for k in keys},
        'target_endpoint_logit_L1_legacy_median_bounds':[statistics.median(r['target_logit_L1_legacy_DT_endpoint_mean_bounds'][j] for r in part) for j in (0,1)],
        'target_endpoint_logit_L1_common_median_bounds':[statistics.median(r['target_logit_L1_common_DT_endpoint_mean_bounds'][j] for r in part) for j in (0,1)]})
usable=[r for r in rows if 'approx_count_relative_error_to_inferred_midpoint' in r]
largest=sorted(usable,key=lambda r:abs(r['approx_count_relative_error_to_inferred_midpoint']),reverse=True)[:10]
report={'observed_local':datetime.now().astimezone().isoformat(),'scope':__doc__,'model_GPU_calls':0,
    'rows':rows,'windows':windows,'native81_count_calibration':raw['rows'][80]['native81_count_calibration'],
    'largest_aggregate_count_discrepancies':[{k:r[k] for k in ['step','approx_count_relative_error_to_inferred_midpoint','total_policy_count_approx','inferred_count_interval']} for r in largest],
    'selected_positive_artifact':raw['selected_exact_positive_artifact'],
    'limits':raw['limits']+['Aggregate count agreement cannot prove each historical B4 denominator exact.',
        'Intervals based only on joint target NLL are broad. Temporal parameter-gradient growth cannot be established by equating an output-L1 bound ratio to a parameter norm ratio.'],
    'rootcause_resolved':False,'accepted_repair':None,'deployed_repair':None}
path=HERE/'historical-B4-weight-analysis-v2.json'
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(path.resolve()),'windows':windows,'largest_count_discrepancies':report['largest_aggregate_count_discrepancies']}))
