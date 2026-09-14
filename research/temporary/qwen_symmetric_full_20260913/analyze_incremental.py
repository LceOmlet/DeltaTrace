"""Export complete task metrics and verify recovery directly from saved vectors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();base=a.run
    runtime=base/'runtime'
    sys.path.insert(0,str(runtime/'experiments/qwen35_comparison'))
    from paper_recovery_common import score_recovery
    plan=json.loads((base/'protocol.json').read_bytes())
    records=[]; costs=[]; receipts=[]; missing=[]; expected_total=0
    def record(family,task,index,metric,dt,ft,scope,fraction=None,unit=None,ft_method='FT_K1'):
        assert np.isfinite([dt,ft]).all()
        records.append(dict(model=family,dataset=task,index=index,metric=metric,DT=dt,FT=ft,
            difference_DT_minus_FT=dt-ft,scope=scope,budget_fraction=fraction,budget_unit=unit,FT_method=ft_method))
    for family in ('qwen3','qwen35'):
        folder=base/(family+'_main_v2'); path=folder/'results.json'
        expected_total+=plan[family]['cases']+448
        if path.exists():
            d=json.loads(path.read_bytes())
            if not a.allow_partial:assert d['status']=='complete'
            expected={(t,i) for t,n in plan[family]['tasks'].items() for i in range(n)}
            rows=[r for r in d['cases'] if r['status']=='complete']
            actual={(r['dataset'],r['index']) for r in rows}
            assert len(rows)==len(actual) and actual<=expected
            if not a.allow_partial:assert actual==expected
            assert d['attribution_profile']==plan[family]['profile'] and d['generation_calls']==0
            if d['status']=='complete':assert sha(folder/'vectors.npz')==d['vectors_sha256']
            for r in rows:
                task,index=r['dataset'],r['index'];dt=r['metrics']['DT'];ft=r['metrics']['FT_K1']
                if family=='qwen3':
                    detail=r['DT_details'];assert detail['pv_rule']=='layer_symmetric'
                    assert len(detail['finite_attention_activity'])==72
                    assert all(x['calls_enqueued']==1 for x in detail['finite_attention_activity'])
                assert dt['views']==dict(rise='signed',mas='positive_part',needle='positive_part')
                for name in ('rise','mas','rise_plus_ap'):
                    record(family,task,index,name,dt[name],ft[name],'released_full_response')
                if task.startswith('niah_'):
                    record(family,task,index,'recall',dt['needle'],r['FT_K3_needle'],
                           'released_NIAH',.1,'released_eligible_tokens','FT_K3')
            for row in d['costs']:
                if row['status']=='returned':costs.append(dict(model=family,scope='main',**row))
            receipts.append(dict(family=family,scope='main',cases=len(rows),results_sha256=sha(path)))
            if actual!=expected:missing.append(dict(model=family,scope='main',missing=len(expected-actual)))
        else:missing.append(dict(model=family,scope='main',missing=plan[family]['cases']))
        folder=base/(family+'_recovery_v2'); path=folder/'status.json'
        if not path.exists():
            missing.append(dict(model=family,scope='paper_recovery',missing=448));continue
        state=json.loads(path.read_bytes());prepared=json.loads((base/(family+'_prepared_recovery.json')).read_bytes())
        assert state['prepared_sha256']==sha(base/(family+'_prepared_recovery.json'))
        expected={(r['dataset'],r['index']):r for r in prepared['cases']}
        actual={(r['dataset'],r['index']) for r in state['cases']}
        assert len(actual)==len(state['cases']) and actual<=expected.keys()
        if not a.allow_partial:assert state['status']=='complete' and len(actual)==448
        for meta in state['cases']:
            path=folder/meta['path']/'results.json';vp=path.with_name('vectors.npz')
            assert sha(path)==meta['results_sha256'] and sha(vp)==meta['vectors_sha256']
            r=json.loads(path.read_bytes());row=expected[r['dataset'],r['index']]
            assert r['status']=='complete' and r['input_sha256']==row['input_sha256']
            assert r['target_weights']==r['FT_target_weights']==row['target_weights']
            with np.load(vp,allow_pickle=False) as v:
                for method in ('DT','FT_K3','FT_K1'):
                    scores=v[method+'_prompt']
                    if method=='DT':assert np.array_equal(scores,v['DT_signed_full'][row['user_positions']])
                    assert r['metrics'][method]==score_recovery(row,scores,r['metrics'][method]['fraction'])
                    for point in r['budgets'][method]:
                        assert point==score_recovery(row,scores,point['fraction'])
            for ft_method in ('FT_K3','FT_K1'):
                dt,ft=r['metrics']['DT'],r['metrics'][ft_method]
                record(family,r['dataset'],r['index'],'recall',dt['recall'],ft['recall'],
                       'paper_recovery',dt['fraction'],dt['budget_unit'],ft_method)
                for dt,ft in zip(r['budgets']['DT'],r['budgets'][ft_method]):
                    record(family,r['dataset'],r['index'],'recall',dt['recall'],ft['recall'],
                           'paper_recovery_budget_curve',dt['fraction'],dt['budget_unit'],ft_method)
            for method,c in r['costs'].items():costs.append(dict(model=family,scope='paper_recovery',
                name=f"{r['dataset']}_{r['index']}_{method}",**c))
        receipts.append(dict(family=family,scope='paper_recovery',cases=len(actual),results_sha256=sha(folder/'status.json')))
        if len(actual)!=448:missing.append(dict(model=family,scope='paper_recovery',missing=448-len(actual)))
    if not a.allow_partial:assert not missing
    groups={}
    for row in records:
        key=tuple(row[k] for k in ('model','dataset','metric','scope','budget_fraction','budget_unit','FT_method'))
        groups.setdefault(key,[]).append(row)
    tables=[];rng=np.random.default_rng(20260913)
    for key,rows in sorted(groups.items(),key=lambda item:str(item[0])):
        model,task,metric,scope,fraction,unit,ft_method=key
        dt=np.array([r['DT'] for r in rows]);ft=np.array([r['FT'] for r in rows]);delta=dt-ft
        expected=plan[model]['tasks'].get(task,100 if task.startswith('vt_') else 48)
        is_complete=len(rows)==expected
        interval=np.quantile(delta[rng.integers(0,len(rows),(5000,len(rows)))].mean(1),[.025,.975]) if is_complete else [None,None]
        tables.append(dict(model=model,dataset=task,metric=metric,scope=scope,budget_fraction=fraction,
            budget_unit=unit,FT_method=ft_method,cases=len(rows),expected_cases=expected,complete_task=is_complete,
            DT=float(dt.mean()),FT=float(ft.mean()),difference_DT_minus_FT=float(delta.mean()),
            difference_ci95_low=interval[0],difference_ci95_high=interval[1]))
    a.output.mkdir(parents=True,exist_ok=True)
    def write_csv(name,rows):
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            with (a.output/name).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_csv('per_case.csv',records);write_csv('task_metrics.csv',tables);write_csv('actual_run_costs.csv',costs)
    report=dict(status='complete' if not missing else 'partial',receipts=receipts,missing=missing,
                expected_model_case_protocol_evaluations=expected_total,
                intervals='Within-task paired descriptive bootstrap, 5000 draws, seed 20260913; no new holdout claim.',
                timing_scope='Actual one-pass execution including compilation; not a repeated warm benchmark.',
                paper_edits=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
