"""Analyze existing per-iteration quantities; association alone is not causality."""
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'collect_temporal_reports_v1.json').read_bytes())['report']
readouts=list(raw['raw_reports'].values())
assert len(readouts)==2 and all(len(r)==85 for r in readouts)
rows=[]
for i,m in enumerate(raw['metrics']):
    assert m['step']==i+1
    groups={}
    for rank in readouts:
        for k,v in rank[i]['value']['raw_advantage_groups'].items():
            g=groups.setdefault(k,{'tokens':0,'sum':0.,'sumsq':0.,'min':math.inf,'max':-math.inf})
            for key in ('tokens','sum','sumsq'):g[key]+=v[key]
            if v['tokens']:
                g['min']=min(g['min'],v['min']);g['max']=max(g['max'],v['max'])
    for g in groups.values():
        if g['tokens']:
            g['mean']=g['sum']/g['tokens'];g['rms']=(g['sumsq']/g['tokens'])**.5
    ns=sum(g['tokens'] for g in groups.values())
    rows.append({'step':m['step'],'metrics':m['values'],'groups':groups,
                 'nonzero_request_policy_tokens':ns,
                 'nonzero_request_target_fraction':groups['self_target']['tokens']/ns,
                 'nonzero_request_raw_mean':sum(g['sum'] for g in groups.values())/ns,
                 'nonzero_request_raw_rms':(sum(g['sumsq'] for g in groups.values())/ns)**.5,
                 'source_paths':[m['sources'],*[r[i]['source'] for r in readouts]]})
windows={}
for a,b in [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,79),(80,85)]:
    part=rows[a-1:b]
    windows[f'{a}-{b}']={'gradient_mean':statistics.mean(r['metrics']['actor/grad_norm'] for r in part),
        'gradient_median':statistics.median(r['metrics']['actor/grad_norm'] for r in part),
        'reward_mean':statistics.mean(r['metrics']['episode/reward/mean'] for r in part),
        'response_length_mean':statistics.mean(r['metrics']['response_length/mean'] for r in part),
        'source_raw_rms_median':statistics.median(r['groups']['prior_source']['rms'] for r in part),
        'source_raw_min':min(r['groups']['prior_source']['min'] for r in part),
        'source_raw_max':max(r['groups']['prior_source']['max'] for r in part),
        'target_fraction_median':statistics.median(r['nonzero_request_target_fraction'] for r in part)}
out={'scope':__doc__,'rows':rows,'windows':windows,
     'limits':['Readout groups cover nonzero-reward DT requests with native padding duplicates; they are not full256 whitened coefficients.',
               'Response_length includes observations; it is not pure generated policy-token count.',
               'No fitted trend or correlation is used as proof of root cause.'],
     'source':str(HERE/'collect_temporal_reports_v1.json')}
(HERE/'temporal-native-analysis-v1.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(windows,indent=2))
