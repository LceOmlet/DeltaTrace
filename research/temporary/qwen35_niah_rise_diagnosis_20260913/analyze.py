"""Exploratory CPU diagnosis of preserved NIAH curves; no metric or model changes."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[4]
SOURCE = BASE / 'audit/qwen35_existing_runtime_20260913/full_dynamic_with_ifr'
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(20260913)

def corr(x, y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None

def auc(y):
    return float((sum(y)-y[0]/2-y[-1]/2)/(len(y)-1))

rows=[]; summaries=[]; sources=[]
for path in sorted(SOURCE.glob('niah_*/results.json')):
    task=path.parent.name
    data=json.loads(path.read_bytes())
    sources.append(dict(task=task,results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        vectors_sha256=hashlib.sha256(path.with_name('vectors.npz').read_bytes()).hexdigest()))
    vectors=np.load(path.with_name('vectors.npz'),allow_pickle=False)
    curves=[]
    for case in data['cases']:
        m=case['metrics']; dt=m.get('DT_signed',m['DT_positive']); ft=m['FT_K1']
        y=np.array(dt['normalized_model_response']); yf=np.array(ft['normalized_model_response'])
        assert abs(auc(y)-m['DT']['rise'])<1e-12
        assert abs(auc(yf)-ft['rise'])<1e-12
        assert dt['scores'][0]==ft['scores'][0] and dt['scores'][-1]==ft['scores'][-1]
        keep=np.array(case['keep']); positions=np.array(case['user_positions'])
        signed=vectors[f'{task}_{case["index"]}_DT_signed_full'][positions]
        gold=set(case['gold']) & set(keep)
        details=case['DT_details']
        relative=details['relative_residual']
        first_zero=int(np.flatnonzero(y==0)[0])
        first_zero_ft=int(np.flatnonzero(yf==0)[0])
        d=dict(task=task,index=case['index'],dt_rise=m['DT']['rise'],ft_rise=ft['rise'],
            delta=m['DT']['rise']-ft['rise'],dt_positive_rise=m['DT_positive']['rise'],
            signed_minus_positive=m['DT']['rise']-m['DT_positive']['rise'],
            dt_recall=m['DT']['needle'],ft_k3_recall=case['FT_K3_needle'],
            dt_mas=m['DT']['mas'],ft_mas=ft['mas'],prompt_length=case['prompt_length'],
            target_length=case['target_length'],eligible_count=len(keep),gold_count=len(gold),
            positive_fraction=float(np.mean(signed[keep]>0)),negative_fraction=float(np.mean(signed[keep]<0)),
            gold_positive_fraction=float(np.mean(signed[list(gold)]>0)) if gold else None,
            root_effect=details['root_effect'],relative_residual=relative,
            max_replay_l2=max(x['replay_relative_L2'] for x in details['layers'].values()),
            score_full=dt['scores'][0],score_deleted=dt['scores'][-1],
            endpoint_gap=dt['scores'][0]-dt['scores'][-1],dt_zero_step=first_zero,ft_zero_step=first_zero_ft)
        for step in (1,2,4,10):
            dg=set(dt['deleted_user_indices'][step]);fg=set(ft['deleted_user_indices'][step])
            d[f'dt_y_{step}']=float(y[step]);d[f'ft_y_{step}']=float(yf[step])
            d[f'dt_deleted_gold_{step}']=len(dg & gold)/len(gold) if gold else None
            d[f'ft_deleted_gold_{step}']=len(fg & gold)/len(gold) if gold else None
            d[f'dt_negative_deleted_{step}']=int(sum(signed[j]<0 for j in dg))
        rows.append(d);curves.append((y,yf))
    vectors.close()
    rr=[r for r in rows if r['task']==task]
    get=lambda k:np.array([r[k] for r in rr],dtype=float)
    dif=get('delta');bootstrap=dif[rng.integers(0,len(rr),(10000,len(rr)))].mean(1)
    sd=np.sort(dif)
    yy=np.array(curves)
    segments={f'{a*5}-{b*5}%':float(np.mean(np.sum((yy[:,0,a:b]-yy[:,1,a:b]+yy[:,0,a+1:b+1]-yy[:,1,a+1:b+1])/40,axis=1))) for a,b in [(0,2),(2,5),(5,10),(10,20)]}
    summary=dict(task=task,n=len(rr),dt_rise=float(get('dt_rise').mean()),ft_rise=float(get('ft_rise').mean()),
        delta_mean=float(dif.mean()),paired_bootstrap_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        delta_median=float(np.median(dif)),dt_wins=int(sum(dif < -1e-12)),ft_wins=int(sum(dif>1e-12)),
        ties=int(sum(np.abs(dif)<=1e-12)),trim10_delta_mean=float(sd[10:-10].mean()),
        signed_changes=int(sum(np.abs(get('signed_minus_positive'))>1e-12)),
        signed_minus_positive_mean=float(get('signed_minus_positive').mean()),
        negative_endpoint_gaps=int(sum(get('endpoint_gap')<=0)),
        mean_positive_fraction=float(get('positive_fraction').mean()),
        mean_gold_positive_fraction=float(get('gold_positive_fraction').mean()),
        mean_target_length=float(get('target_length').mean()),
        abs_relative_residual_median=float(np.median(np.abs(get('relative_residual')))),
        abs_relative_residual_p95=float(np.quantile(np.abs(get('relative_residual')),.95)),
        max_replay_l2=float(get('max_replay_l2').max()),
        correlation_delta_target_length=corr(dif,get('target_length')),
        correlation_delta_abs_residual=corr(dif,np.abs(get('relative_residual'))),
        mean_curves=dict(dt=yy[:,0].mean(0).tolist(),ft=yy[:,1].mean(0).tolist()),
        delta_auc_segments=segments,
        worst_cases=[{k:r[k] for k in ('index','dt_rise','ft_rise','delta','target_length','dt_recall','ft_k3_recall','relative_residual')} for r in sorted(rr,key=lambda r:r['delta'],reverse=True)[:8]])
    summaries.append(summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('mean_curves','worst_cases')},ensure_ascii=False))
assert len(rows)==600
with (OUT/'cases.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'diagnosis.json').write_text(json.dumps(dict(status='exploratory_saved_curve_analysis',
    model_calls=0,bootstrap_seed=20260913,bootstrap_samples=10000,
    inference_limit='Paired case-resampling confidence intervals are exploratory, unadjusted for six comparisons. Associations do not identify propagation-rule causality.',
    sources=sources,tasks=summaries),indent=2)+'\n',encoding='utf-8')
