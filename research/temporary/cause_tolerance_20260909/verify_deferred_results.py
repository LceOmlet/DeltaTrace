"""Verify saved full vectors, source identities, complete cost and guard records.

No model or metric calls. Equality is an observation for scheduling-only changes,
not a general precision requirement on FA/FLA or future tolerance candidates.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()


def read_job(name):
    path=HERE/name
    result=json.loads((path/'results.json').read_bytes())
    assert result['status']=='complete'
    assert sha((path/'vectors.npz').read_bytes())==result['vectors_sha256']
    vectors=np.load(path/'vectors.npz',allow_pickle=False)
    assert all(np.isfinite(vectors[n]).all() for n in vectors.files)
    return path,result,vectors


def main():
    summaries={}
    for name in ('qwen3_deferred_pilot','qwen3_deferred16','qwen35_deferred_pilot','qwen35_deferred16'):
        path,r,z=read_job(name)
        driver='benchmark_'+name.replace('_pilot','')+'.py'
        assert sha((HERE/driver).read_bytes())==r.get('script_sha256',r.get('driver_sha256'))
        assert r['generation_calls']==0 and r.get('FT_calls',0)==0 and r.get('metric_root_calls',0)==0
        q3=name.startswith('qwen3_');batch=name=='qwen35_deferred16'
        rows=[];measured=[]
        groups=r['batches'] if batch else [{'cases':[c['case']]} for c in r['comparisons']]
        for i,group in enumerate(groups):
            for case in group['cases']:
                keys=[k for k in z.files if (k.endswith('/'+case) if batch else k.startswith(case+'_'))]
                assert len(keys)==6 and all(np.array_equal(z[keys[0]],z[k]) for k in keys)
            times=[];peaks=[];cold=[]
            modes=('baseline','accelerated' if batch else 'candidate' if q3 else 'deferred')
            for mode in modes:
                if batch:
                    calls=[c for c in r['calls'] if c['name'].startswith('measured_') and '_'+mode+'/' in c['name'] and c['cases']==group['cases']]
                    warms=[c for c in r['calls'] if c['name']==f'warm_{mode}/{i}']
                else:
                    prefix=group['cases'][0]+'_'+mode
                    calls=[c for c in r['calls'] if c['name'].startswith(prefix+'_measured')]
                    warms=[c for c in r['calls'] if c['name']==prefix+'_cold']
                assert len(calls)==2 and len(warms)==1 and all(c['status']=='returned' for c in calls+warms)
                measured.extend(calls)
                times.append(statistics.mean(c['seconds'] for c in calls))
                peaks.append(max(max(c['peak_allocated'],c.get('details',{}).get('root_peak_allocated',0)) for c in calls))
                cold.append(warms[0]['seconds'])
                if batch:
                    for c in calls+warms:
                        assert c['actual_root']==group['actual_root']
                        assert c['actual_root']['sample_batch']==2 and c['actual_root']['endpoint_batch']==4
                        if c in calls:assert c['compiler_before']==c['compiler_after']
                        if mode=='accelerated':assert c['details']['controller_diagnostic_scheduling']['all_32_finite_checks_passed']
                elif q3 and mode=='candidate':
                    for c in calls+warms:
                        v=c['stages']['validation'];assert v['all_passed'] and v['predicates']==828 and v['statistics']==36
            rows.append({'cases':','.join(group['cases']),'baseline_seconds':times[0],'deferred_seconds':times[1],
                         'reduction_fraction':1-times[1]/times[0],'baseline_peak_bytes':peaks[0],'deferred_peak_bytes':peaks[1],
                         'baseline_first_shape_seconds':cold[0],'deferred_first_shape_seconds':cold[1],
                         'all_six_vectors_equal':True})
        totals=[sum(x[k] for x in rows) for k in ('baseline_seconds','deferred_seconds')]
        summary={'status':'verified','result_sha256':sha((path/'results.json').read_bytes()),'vectors_sha256':r['vectors_sha256'],
                 'cases':sum(len(g['cases']) for g in groups),'full_DT_calls':len(z.files)//(2 if batch else 1),
                 'sample_batch':2 if batch else 1,'endpoint_batch':4 if batch else 2,'equal_vectors':len(z.files),
                 'baseline_seconds_per_pass':totals[0],'deferred_seconds_per_pass':totals[1],
                 'reduction_fraction':1-totals[1]/totals[0],
                 'peak_bytes':{m:max(row[k] for row in rows) for m,k in [('baseline','baseline_peak_bytes'),('deferred','deferred_peak_bytes')]},
                 'rows':rows,'quality_statement':'All complete vectors match within the paired process, so signed RISE ordering and positive MAS/needle inputs are identical. No new metric evaluations or quality improvement claim.',
                 'timing_scope':'Mean of two interleaved complete warmed calls per mode; cold/first-shape calls separate. Not cross-process FT timing.'}
        if q3:
            assert r['delayed_guard_checks']=={'valid':{'raised_before_return':False,'expected':False},'nonfinite':{'raised_before_return':True,'expected':True}}
            summary['invalid_value_rejected_before_return']=True
            for n,digest in r['candidate_sources'].items():assert sha((HERE/n).read_bytes())==digest
        else:assert sha((HERE/'qwen35_deferred_controller.py').read_bytes())==r['deferred_controller_sha256']
        (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',newline='\n')
        with (path/'costs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summaries[name]=summary
    for family in ('qwen3','qwen35'):
        path,r,z=read_job('profile_'+family)
        assert sha((HERE/'profile_native_dt.py').read_bytes())==r['script_sha256']
        assert len(r['calls'])==4 and all(c['vector_equal_to_warm'] for c in r['calls'] if c['mode']=='profile')
        selected=[{'case':c['case'],'synchronizations':[e for e in c['events'] if e['key'] in ('mcStreamSynchronize','mcDeviceSynchronize')]} for c in r['calls'] if c['mode']=='profile']
        (path/'summary.json').write_text(json.dumps({'status':'verified','events':selected,'limitation':'Profiler timing includes overhead. Inclusive event times overlap and are not summed into runtime fractions.'},indent=2)+'\n',newline='\n')
    manifest=json.loads((ROOT/'deltatrace/accelerated/deferred_sources.json').read_bytes())
    for n,digest in manifest['files'].items():assert sha((ROOT/n).read_bytes())==digest
    for n in ('deferred_validation.py','qwen3_deferred_pair.py','qwen3_deferred_finite.py','qwen3_deferred_replay.py'):
        assert (ROOT/'deltatrace/accelerated/qwen3'/n).read_bytes()==(HERE/n).read_bytes()
    assert (ROOT/'deltatrace/accelerated/qwen35/controller_deferred.py').read_bytes()==(HERE/'qwen35_deferred_controller.py').read_bytes()
    print(json.dumps({n:{k:s[k] for k in ('cases','equal_vectors','baseline_seconds_per_pass','deferred_seconds_per_pass','reduction_fraction','peak_bytes')} for n,s in summaries.items()},indent=2))


if __name__=='__main__':main()
