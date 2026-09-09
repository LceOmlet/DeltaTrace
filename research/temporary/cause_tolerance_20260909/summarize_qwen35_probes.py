"""Audit real-gate records and the scalar-reuse screen without model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--snapshots',type=Path,required=True)
args = p.parse_args()
sha = lambda b: hashlib.sha256(b).hexdigest()
root = args.directory
baseline = {r['dataset']+'_'+r['index']:r for r in csv.DictReader((root/'baseline_cases.csv').open()) if r['family']=='qwen35'}
gate = json.loads((root/'qwen35_real_gates/results.json').read_bytes())
assert gate['status']=='complete' and len(gate['cases'])==16 and gate['root_calls']==17
assert gate['metric_calls']==gate['generation_calls']==0
assert sha((root/'qwen35_real_gates/vectors.npz').read_bytes())==gate['vectors_sha256']
assert sha((root/'inspect_qwen35_real_gates.py').read_bytes())==gate['script_sha256']
vectors = np.load(root/'qwen35_real_gates/vectors.npz')
assert len(vectors.files)==16*(24*5+1)
old = {}
for folder,report_sha,npz_sha in [
        ('codex_clean_development16_20260909_v1','04c59aaee006b49bb1c93d5ded737805b17570c3517aa045370bd38d37226cdb','9f450bd4e5c07c2ebcf14c8f114f5bd7aed814db3413b66f1562bc6b28943fec'),
        ('codex_clean_development16_20260909_mh_recovery_v1','5999968354f16fe3760a4e2f9bfae9401524d366648cfe7f11fe563339719467','6396709205613b34aaa7fb05d61304c3b9c968b6df31e13cd10e3341453efed7')]:
    path=args.snapshots/'tmp'/folder/'qwen35'
    assert sha((path/'results.json').read_bytes())==report_sha
    assert sha((path/'vectors.npz').read_bytes())==npz_sha
    old_vectors=np.load(path/'vectors.npz')
    for row in json.loads((path/'results.json').read_bytes())['cases']:
        if row['status']=='complete':
            name=row['dataset']+'_'+str(row['index'])
            old[name]=(row,old_vectors[name+'_DT_signed_full'])
summary={'status':'complete','raw_sha256':sha((root/'qwen35_real_gates/results.json').read_bytes()),
         'vectors_sha256':gate['vectors_sha256'],'input_identities_verified':16,'GDN_layers_verified':384,
         'tasks':{},'cross_run_comparisons':[],
         'interpretation':'Conditions for baseline-prefix suppression occur in both tasks at similar rates. This does not establish a cause of the NI-specific DT/FT gap. No method repair accepted.'}
for c in gate['cases']:
    assert c['input_sha256']==baseline[c['case']]['input_sha256']
    assert len(c['layers'])==24 and len({l['layer'] for l in c['layers']})==24
    a=old[c['case']][1];b=vectors[c['case']+'_signed']
    assert np.isfinite(b).all() and a.shape==b.shape
    summary['cross_run_comparisons'].append({'case':c['case'],'relative_l2':float(np.linalg.norm(b-a)/np.linalg.norm(a)),
        'old_root_effect':old[c['case']][0]['DT_details']['root_effect'],'new_root_effect':c['root_effect']})
for task in ('niah_mq_q2','morehopqa'):
    layers=[l for c in gate['cases'] if c['case'].startswith(task+'_') for l in c['layers']]
    out={'layers':len(layers),'windows':{}}
    assert len(layers)==192
    for lag in gate['lags']:
        rows=[l['windows'][str(lag)] for l in layers]
        totals={k:sum(r[k] for r in rows) for k in ('count','actual_surviving','baseline_suppressed','actual_suppressed_reverse')}
        assert 0<=totals['baseline_suppressed']<=totals['actual_surviving']<=totals['count']
        out['windows'][str(lag)]={**totals,
            'suppressed_fraction_all':totals['baseline_suppressed']/totals['count'],
            'suppressed_fraction_among_actual_surviving':totals['baseline_suppressed']/max(totals['actual_surviving'],1)}
    masses={k:sum(l['branch_mass'][k]['positive_mass']-l['branch_mass'][k]['negative_mass'] for l in layers)
            for k in ('q','k','v','beta','g')}
    out['branch_pool_abs_fractions']={k:v/sum(masses.values()) for k,v in masses.items()}
    out['branch_pool_scope']='Pooled intermediate branch mass, double-counts across layers; not fractions of final output or causal deletion effects.'
    summary['tasks'][task]=out
(root/'qwen35_real_gates/summary.json').write_text(json.dumps(summary,indent=2)+'\n')

path=root/'qwen35_scalar_reuse'
r=json.loads((path/'results.json').read_bytes())
assert r['status']=='complete' and len(r['root_calls'])==12 and not r['metrics_recomputed']
assert sha((path/'vectors.npz').read_bytes())==r['vectors_sha256']
assert sha((root/'benchmark_qwen35_scalar_reuse.py').read_bytes())==r['script_sha256']
v=np.load(path/'vectors.npz')
assert len(v.files)==12
out={'status':'complete','raw_sha256':sha((path/'results.json').read_bytes()),
     'vectors_sha256':r['vectors_sha256'],'cases':[],'comparisons':r['comparisons']}
for row in r['cases']:
    name=row['dataset']+'_'+str(row['index'])
    assert row['input_sha256']==baseline[name]['input_sha256']
    for mode in ('baseline','reuse'):
        names=[c['name'] for c in r['calls'] if c.get('warm') and c['name'].startswith(name+'_'+mode+'_')]
        assert len(names)==2 and np.array_equal(v[names[0]],v[names[1]])
    record={'case':name}
    for mode in ('baseline','reuse'):
        calls=[c for c in r['calls'] if c.get('warm') and c['name'].startswith(name+'_'+mode+'_')]
        record[mode]={'mean_seconds':statistics.mean(c['seconds'] for c in calls),
                      'each_seconds':[c['seconds'] for c in calls],
                      'peak_allocated':max(c['peak_allocated'] for c in calls)}
    a=v[name+'_baseline_measured5'];b=v[name+'_reuse_measured4']
    item=next(c for c in r['comparisons'] if c['case']==name)
    assert abs(item['relative_l2']-float(np.linalg.norm(b-a)/np.linalg.norm(a)))<1e-12
    out['cases'].append(record)
reduction=1-sum(c['reuse']['mean_seconds'] for c in out['cases'])/sum(c['baseline']['mean_seconds'] for c in out['cases'])
assert abs(reduction-r['mean_latency_reduction_fraction'])<1e-12
out.update(mean_latency_reduction_fraction=reduction,quality_followup_warranted=reduction>=.03,
           disposition='Stopped at cost screen: no aggregate speed or peak-memory benefit; no quality acceptance or metric claim.')
(path/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'gate_cases':16,'gate_layers':384,'reuse_cases':out['cases'],
                  'reuse_latency_reduction':reduction,'reuse_disposition':out['disposition']},indent=2))
