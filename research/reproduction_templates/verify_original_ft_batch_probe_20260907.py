"""Independent original curves, unchanged evaluator, trajectories and batch costs."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_original_ft_batch_probe_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete'
assert p==json.loads((A/'original_ft_batch_probe_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'original_ft_batch_probe_20260907.py')
for name,digest in p['sources'].items():assert sha(F/name)==sha(A/name)==digest
parent_file=A/'snapshot'/p['required_parent'].lstrip('/')
assert sha(parent_file)==p['required_parent_sha256'];parent=json.loads(parent_file.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
for name,digest in p['official_normalized_sources'].items():
    f=A/'snapshot${FLASHTRACE_ROOT}'/name
    assert hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
assert d['dtype']==parent['dtype']=='torch.float16' and d['torch_version']==parent['torch_version']
assert len(d['attempts'])==27 and all(a['complete'] for a in d['attempts'])
for key,value in p['budget'].items():assert d[key]==value
for source,target in [('native_forwards','physical_evaluation_forwards'),('native_forward_trajectories','evaluation_trajectories'),
                      ('native_decoder_layer_calls','native_decoder_layer_calls'),('native_decoder_layer_trajectories','native_decoder_layer_trajectories')]:
    assert sum(a['cost'][source] for a in d['attempts'])==d[target]
for a in d['attempts']:
    n=a['batch_size'] if a['warm'] else 42;c=a['cost'];scheduler=a['scheduler_counts']
    assert c['native_forwards']==math.ceil(n/a['batch_size'])==scheduler['physical_evaluation_forwards']
    assert c['native_forward_trajectories']==n==scheduler['evaluation_trajectories']
    assert sum(scheduler['actual_batch_sizes'])==n and max(scheduler['actual_batch_sizes'])<=a['batch_size']
    assert c['native_decoder_layer_calls']==36*c['native_forwards'] and c['native_decoder_layer_trajectories']==36*n
    assert c['extra_replay_calls']==c['extra_replay_trajectories']==c['vjps']==0
    assert c['seconds']>0 and c['peak_allocated_bytes']>0

def area(x):return float((x.sum()-(x[0]+x[-1])/2)/(len(x)-1))
def metrics(curve,row,method):
    x=np.asarray(curve,dtype=np.float64);w=np.asarray(row['scores'][method],dtype=np.float32)
    keep=row['keep_local_indices'];masks=row['evaluation_masks'][method]
    norm=np.minimum.accumulate(np.clip((x-x[-1])/abs(x[0]-x[-1]),0,1))
    total=float(w[sorted(keep)].sum(dtype=np.float32));density=[1.]
    for step in range(1,21):
        group=sorted(set(masks[step])-set(masks[step-1]))
        density.append(density[-1]-float(w[group].sum(dtype=np.float32))/total if total>0 else 1-step/20)
    corrected=np.clip(norm+abs(norm-np.asarray(density)),0,1)
    corrected=(corrected-corrected.min())/(corrected.max()-corrected.min()) if corrected.max()>corrected.min() else np.linspace(1,0,21)
    return {'rise':area(norm),'mas':area(corrected)}

out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'budget':p['budget'],'cases':[],
 'scope':'Six original frozen P1/FT curves, NI0/NI2/MH0 only. Original evaluator and eager model unchanged; batch scheduling only. This is evaluation batching, not multi-example attribution or independent candidate confirmation.'}
assert [(r['dataset'],r['idx']) for r in d['records']]==[tuple(x) for x in p['selection']]
for r in d['records']:
    old=next(x for x in parent['records'] if (x['dataset'],x['idx'])==(r['dataset'],r['idx']))
    assert r['methods']==['strong_secant_pv_content_P1',p['FT_curve_control'][r['dataset']]]
    assert r['N']==len(old['input_ids']) and r['prompt_len']==old['prompt_len']
    assert r['parent_input_ids_sha256']==hashlib.sha256(json.dumps(old['input_ids']).encode()).hexdigest()
    for m in r['methods']:assert r['original_curves'][m]==old['metrics'][m]['raw_curve']
    assert len(r['runs'])==6 and {(v['repeat'],v['batch_size']) for v in r['runs']}=={(i,b) for i in range(2) for b in [1,2,4]}
    single=[x for x in r['runs'] if x['batch_size']==1]
    assert single[0]['curves']==single[1]['curves']
    baseline=single[0]['curves'];ref_time=statistics.median(x['cost']['seconds'] for x in single)
    case={'dataset':r['dataset'],'idx':r['idx'],'N':r['N'],'batch':{}}
    for b in [1,2,4]:
        runs=[x for x in r['runs'] if x['batch_size']==b];curve_errors=[];rise_errors=[];mas_errors=[];raw_ok=True
        detail={'repeat_curves_identical':runs[0]['curves']==runs[1]['curves'],'methods':{}}
        for m in r['methods']:
            original=np.asarray(r['original_curves'][m]);ref=np.asarray(baseline[m]);now=np.asarray(runs[0]['curves'][m])
            assert len(now)==21 and np.isfinite(now).all()
            reference_metrics=metrics(ref,old,m);current_metrics=metrics(now,old,m)
            # Original parent metric independently reconstructed as well.
            parent_metrics=metrics(original,old,m)
            for k in ['rise','mas']:assert abs(parent_metrics[k]-old['metrics'][m][k])<5e-6
            delta=abs(now-ref);curve_errors.append(float(delta.max()))
            raw_ok &= bool((delta<=np.maximum(.5,.001*abs(ref))).all())
            rise_errors.append(abs(current_metrics['rise']-reference_metrics['rise']))
            mas_errors.append(abs(current_metrics['mas']-reference_metrics['mas']))
            detail['methods'][m]={'curve':now.tolist(),'max_raw_error_to_same_job_B1':float(delta.max()),
                'same_job_B1_exact_to_original_parent':bool(np.array_equal(ref,original)),
                'same_job_B1_max_error_to_original_parent':float(abs(ref-original).max()),
                'metrics':current_metrics,'metrics_change_to_B1':{k:current_metrics[k]-reference_metrics[k] for k in current_metrics}}
        seconds=statistics.median(x['cost']['seconds'] for x in runs)
        detail.update(median_seconds_per_42_states=seconds,trajectories_per_second=42/seconds,speedup_over_B1=ref_time/seconds,
            max_peak_bytes=max(x['cost']['peak_allocated_bytes'] for x in runs),physical_calls_per_42_states=runs[0]['cost']['native_forwards'],
            max_raw_curve_error=max(curve_errors),max_RISE_error=max(rise_errors),max_MAS_error=max(mas_errors),
            numerical_tolerances_pass=raw_ok and max(rise_errors)<=.002 and max(mas_errors)<=.002 and detail['repeat_curves_identical'])
        case['batch'][str(b)]=detail
    out['cases'].append(case)
out['all_cases_numerical_pass']={str(b):all(c['batch'][str(b)]['numerical_tolerances_pass'] for c in out['cases']) for b in [1,2,4]}
(A/'original_ft_batch_probe_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'budget':out['budget'],'all_cases_numerical_pass':out['all_cases_numerical_pass'],
 'costs':[{'dataset':c['dataset'],'idx':c['idx'],'batch':{b:{k:v[k] for k in ['speedup_over_B1','max_peak_bytes','max_raw_curve_error','max_RISE_error','max_MAS_error']} for b,v in c['batch'].items()}} for c in out['cases']]},indent=2))
