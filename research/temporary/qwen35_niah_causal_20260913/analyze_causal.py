"""Analyze completed native intervention cases, including two-factor Shapley effects."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def auc(x):return float((np.sum(x)-x[0]/2-x[-1]/2)/(len(x)-1))
def curve_area(scores,full,deleted):
    gap=abs(full-deleted)
    if gap==0:return None
    return auc(np.minimum.accumulate(np.clip((np.asarray(scores)-deleted)/gap,0,1)))
def mean(x):return float(np.mean(x)) if x else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    rows=[];factors=[];checks=[];credits=[]
    for file in sorted(a.run.glob('niah_*/results.json')):
        c=json.loads(file.read_bytes())
        if c['status']!='complete':continue
        for name in ('vectors','token_logprobs'):
            assert hashlib.sha256(file.with_name(name+'.npz').read_bytes()).hexdigest()==c[name+'_sha256']
        ev=c['evaluations'];curves=c['curves']
        values=lambda label,target:np.array([ev[k][target] for k in curves[label]['evaluations']])
        with np.load(file.with_name('vectors.npz'),allow_pickle=False) as vv:
            attribution=vv['DT_original'].astype(np.float64)
            f11=values('DT_original','full_fp64')[0];f00=values('DT_original','full_fp64')[-1]
            f10=values('DT_original_only_query','full_fp64')[-1]
            f01=values('DT_original_restore_query','full_fp64')[-1]
            exact_q=.5*((f11-f10)+(f01-f00));exact_b=.5*((f11-f01)+(f10-f00))
            assert abs(exact_q+exact_b-(f11-f00))<1e-8
            aq=float(attribution[c['query']].sum());ab=float(attribution[c['nonquery']].sum())
            credits.append(dict(dataset=c['dataset'],index=c['index'],dt_query_credit=aq,dt_nonquery_credit=ab,
                group_shapley_query_credit=float(exact_q),group_shapley_nonquery_credit=float(exact_b),
                native_endpoint_effect=float(f11-f00),group_interaction=float(f11-f10-f01+f00),
                dt_query_fraction=aq/(aq+ab),group_shapley_query_fraction=float(exact_q/(f11-f00))))
        for name,curve in curves.items():
            for target in ('full','answer','answer_content','full_fp64','answer_fp64','answer_content_fp64'):
                scores=values(name,target)
                area=curve_area(scores,scores[0],scores[-1])
                rows.append(dict(dataset=c['dataset'],index=c['index'],method=name,target=target,
                    rise=area,endpoint_gap=float(scores[0]-scores[-1]),mas=curve.get('mas') if target=='full' else None))
                if target=='full':
                    assert area==curve['full_rise']
        for target in ('full','answer','answer_content','full_fp64','answer_fp64','answer_content_fp64'):
            by_method={}
            for name in ('DT_original','FT_answer_K1'):
                both=values(name,target);q=values(name+'_only_query',target);b=values(name+'_restore_query',target)
                assert both[0]==q[0]==b[0]
                s0,sall=both[0],both[-1]
                joint=curve_area(both,s0,sall);only_q=curve_area(q,s0,sall);only_b=curve_area(b,s0,sall)
                empty=curve_area(np.full(len(both),s0),s0,sall)
                if empty is None:continue
                q_effect=.5*((empty-only_q)+(only_b-joint))
                b_effect=.5*((empty-only_b)+(only_q-joint))
                assert abs(q_effect+b_effect-(empty-joint))<1e-12
                by_method[name]=dict(query=q_effect,nonquery=b_effect,area=joint,restore_query_area=only_b,
                    only_query_area=only_q,endpoint_gap=float(s0-sall),empty_area=empty)
            if len(by_method)==2:
                dt,ft=by_method['DT_original'],by_method['FT_answer_K1']
                assert dt['endpoint_gap']==ft['endpoint_gap']
                qdiff=ft['query']-dt['query'];bdiff=ft['nonquery']-dt['nonquery'];gap=dt['area']-ft['area']
                assert abs(qdiff+bdiff-gap)<1e-12
                factors.append(dict(dataset=c['dataset'],index=c['index'],target=target,dt_minus_ft=gap,
                    query_schedule_contribution=qdiff,nonquery_schedule_contribution=bdiff,
                    fixed_original_denominator_restore_query_gap=dt['restore_query_area']-ft['restore_query_area'],
                    dt=dt,ft=ft))
        checks.append(dict(dataset=c['dataset'],index=c['index'],dt_vector=c['DT_archive_comparison'],
            old_dt_rise_difference=curves['DT_original']['archive_rise_difference'],
            old_ft_rise_difference=curves['FT_answer_K1']['archive_rise_difference'],
            old_dt_score_max_difference=curves['DT_original']['archive_max_score_difference'],
            old_ft_score_max_difference=curves['FT_answer_K1']['archive_max_score_difference']))
    groups=[];fgroups=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        for target in ('full','answer','answer_content','full_fp64','answer_fp64','answer_content_fp64'):
            for method in sorted({r['method'] for r in rows}):
                rr=[r for r in rows if (r['dataset'],r['method'],r['target'])==(dataset,method,target)]
                groups.append(dict(dataset=dataset,method=method,target=target,n=len(rr),
                    rise=mean([r['rise'] for r in rr if r['rise'] is not None]),undefined=sum(r['rise'] is None for r in rr),
                    nonpositive_endpoint=sum(r['endpoint_gap']<=0 for r in rr),mas=mean([r['mas'] for r in rr if r['mas'] is not None])))
            rr=[r for r in factors if (r['dataset'],r['target'])==(dataset,target)]
            fgroups.append(dict(dataset=dataset,target=target,n=len(rr),**{k:mean([r[k] for r in rr]) for k in
                ('dt_minus_ft','query_schedule_contribution','nonquery_schedule_contribution','fixed_original_denominator_restore_query_gap')}))
    credit_means=[]
    for dataset in sorted({x['dataset'] for x in credits}):
        rr=[x for x in credits if x['dataset']==dataset]
        credit_means.append(dict(dataset=dataset,n=len(rr),**{k:mean([x[k] for x in rr]) for k in
            ('dt_query_fraction','group_shapley_query_fraction','group_interaction')}))
    out=dict(completed_cases=len(checks),cases=rows,task_means=groups,query_factorization=factors,
        group_credit=credits,group_credit_task_means=credit_means,
        query_factorization_task_means=fgroups,archive_checks=checks,
        interpretation='The Shapley factors are the two native deletion schedules (query and nonquery), using original full/deleted endpoint normalization for all four counterfactual curves. Their contributions add exactly to the observed RISE gap. This attributes effects of the schedules, not a specific internal propagation rule. Changed-scope or changed-target RISEs are separate diagnostic measures.')
    (a.run/'causal_analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    if rows:
        with (a.run/'causal_cases.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(completed_cases=len(checks),full_target_means=[r for r in groups if r['target']=='full'],
        query_factorization=[r for r in fgroups if r['target']=='full'],archive_checks=checks),indent=2))

if __name__=='__main__':main()
