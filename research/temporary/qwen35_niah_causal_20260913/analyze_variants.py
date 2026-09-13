"""Paired quality analysis and fixed selection criteria for uniform-rule candidates."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--causal-run',type=Path)
    a=p.parse_args();status=json.loads((a.run/'status.json').read_bytes());plan=json.loads((a.run/'protocol.json').read_bytes())
    methods=plan['methods'];rows=[];checks=[];rng=np.random.default_rng(20260913)
    for file in sorted(a.run.glob('niah_*/results.json')):
        c=json.loads(file.read_bytes())
        if c['status']!='complete':continue
        for name in ('vectors','token_logprobs'):
            assert hashlib.sha256(file.with_name(name+'.npz').read_bytes()).hexdigest()==c[name+'_sha256']
        q=None
        if a.causal_run:
            ref=a.causal_run/file.parent.name/'results.json'
            if ref.exists():q=json.loads(ref.read_bytes())['query']
        with np.load(file.with_name('vectors.npz'),allow_pickle=False) as vv:
            for method in methods:
                d=c['methods'][method]
                row=dict(dataset=c['dataset'],index=c['index'],method=method,**{k:d[k] for k in ('rise','mas','recall','relative_residual')})
                if q is not None:
                    weights=vv[method].astype(np.float64);row['query_credit_fraction']=float(weights[q].sum()/weights[c['keep']].sum())
                    for step in (2,4):
                        eid=c['curves'][method+'_signed']['evaluations'][step]
                        deleted=c['evaluations'][eid]['deleted'];row[f'query_fraction_deleted_{step}']=len(set(deleted)&set(q))/len(q)
                rows.append(row)
        for method,values in [('FT_K1',c['FT_K1']),('FT_K3',dict(rise=None,mas=None,recall=c['FT_K3_recall']))]:
            rows.append(dict(dataset=c['dataset'],index=c['index'],method=method,**values,relative_residual=None))
        checks.append({k:c.get(k) for k in ('dataset','index','archive_dt_relative_l2','archive_dt_max_abs','archive_ft_rise_difference','archive_dt_rise_difference','archive_dt_mas_difference')})
    aggregates=[];paired=[]
    datasets=sorted({x['dataset'] for x in rows})
    for dataset in datasets+['macro']:
        selected=[x for x in rows if dataset=='macro' or x['dataset']==dataset]
        for method in methods+['FT_K1','FT_K3']:
            rr=[x for x in selected if x['method']==method]
            item=dict(dataset=dataset,method=method,n=len(rr))
            for metric in ('rise','mas','recall','query_credit_fraction','query_fraction_deleted_2','query_fraction_deleted_4'):
                values=[x[metric] for x in rr if x.get(metric) is not None]
                item[metric]=float(np.mean(values)) if values else None
            aggregates.append(item)
        original={(x['dataset'],x['index']):x for x in selected if x['method']=='DT_original'}
        for method in methods:
            if method=='DT_original':continue
            rr=[x for x in selected if x['method']==method]
            for metric in ('rise','mas','recall'):
                difference=np.array([x[metric]-original[x['dataset'],x['index']][metric] for x in rr])
                if not len(difference):continue
                bootstrap=difference[rng.integers(0,len(difference),(10000,len(difference)))].mean(1)
                paired.append(dict(dataset=dataset,method=method,metric=metric,n=len(rr),delta=float(difference.mean()),
                    ci95=np.quantile(bootstrap,[.025,.975]).tolist(),better=int(sum(difference<0 if metric!='recall' else difference>0))))
    eligible=[]
    if len(checks)>0:
        by={(x['dataset'],x['method']):x for x in aggregates};base=by['macro','DT_original']
        for method in methods:
            if method=='DT_original':continue
            x=by['macro',method]
            mq_better=all(by[task,method]['rise']<by[task,'DT_original']['rise'] for task in datasets if task.startswith('niah_mq'))
            if mq_better and x['rise']<base['rise'] and x['mas']<=base['mas']+.01 and x['recall']>=base['recall']-.02:
                eligible.append(method)
        eligible.sort(key=lambda m:by['macro',m]['rise'])
    out=dict(status=status['status'],completed_cases=len(checks),rows=rows,means=aggregates,paired=paired,archive_checks=checks,
        candidates_meeting_screen_criteria=eligible,provisional_best=eligible[0] if eligible else None,
        inference='Bootstrap intervals are exploratory, not corrected for multiple candidate/metric comparisons. Candidate selection uses this screen; only separately constructed validation inputs can test the frozen choice.')
    (a.run/'variant_analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    with (a.run/'variant_means.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregates[0]) if aggregates else ['dataset']);w.writeheader();w.writerows(aggregates)
    print(json.dumps(dict(status=status['status'],completed_cases=len(checks),eligible=eligible,
        means=[{k:r[k] for k in ('dataset','method','n','rise','mas','recall')} for r in aggregates if r['dataset']=='macro']),indent=2))

if __name__=='__main__':main()
