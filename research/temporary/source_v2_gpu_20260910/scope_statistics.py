"""Prespecified paired Recall comparisons; never selects a new candidate."""
import numpy as np

from analyze import paired_bootstrap, interval

TASKS=['vt_h2_c3','vt_h4_c1','vt_h6_c1','vt_h10_c1','hotpotqa_long']
FRACTIONS=[.05,.1,.2]
VIEWS=['raw','density']


def summarize_scope(rows, split):
    table={(x['dataset'],x['index'],x['method'],x['view'],x['fraction']):x for x in rows}
    columns=[(view,f) for view in VIEWS for f in FRACTIONS]
    task_stats={};boots={};points={}
    for t in TASKS:
        indices=split['tasks'][t]['validation']
        assert len(indices)==16
        diffs=np.asarray([[table[t,i,'DT_target',v,f]['recall']-table[t,i,'FT_K3',v,f]['recall']
            for v,f in columns] for i in indices])
        boot=paired_bootstrap(diffs,np.random.default_rng(np.random.SeedSequence(73,spawn_key=(TASKS.index(t),))))
        boots[t]=boot;points[t]=diffs.mean(axis=0)
        task_stats[t]={}
        for j,(view,f) in enumerate(columns):
            methods={m:float(np.mean([table[t,i,m,view,f]['recall'] for i in indices])) for m in ('DT_target','FT_K1','FT_K3')}
            task_stats[t].setdefault(view,{})[str(f)]=dict(methods=methods,**interval(diffs[:,j],boot[:,j]),
                joint_exact_ceiling=all(table[t,i,m,view,f]['recall']==table[t,i,m,view,f]['ceiling'] for i in indices for m in ('DT_target','FT_K3')),
                mean_ceiling=float(np.mean([table[t,i,'DT_target',view,f]['ceiling'] for i in indices])))
    groups={};primary={}
    for group,members in [('VT',TASKS[:4]),('HotpotQA',TASKS[4:]),('all_five_tasks',TASKS)]:
        boot=np.mean([boots[t] for t in members],axis=0)
        mean=np.mean([points[t] for t in members],axis=0)
        groups[group]={}
        for j,(view,f) in enumerate(columns):
            methods={m:float(np.mean([task_stats[t][view][str(f)]['methods'][m] for t in members])) for m in ('DT_target','FT_K1','FT_K3')}
            value=dict(methods=methods,mean_difference=float(mean[j]),ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),
                joint_exact_ceiling=all(task_stats[t][view][str(f)]['joint_exact_ceiling'] for t in members))
            groups[group].setdefault(view,{})[str(f)]=value
            if group!='all_five_tasks' and f==.1:
                lo,hi=np.quantile(boot[:,j],[.00625,.99375]).tolist()
                primary[group+'_'+view]=dict(value,adjusted_ci9875=[lo,hi],
                    adjusted_verdict='advantage' if lo>0 else 'disadvantage' if hi<0 else 'ceiling_parity' if value['joint_exact_ceiling'] else 'inconclusive')
    assert set(primary)=={'VT_raw','VT_density','HotpotQA_raw','HotpotQA_density'}
    return dict(task_comparisons=task_stats,groups=groups,primary_comparisons=primary,
        primary_family=dict(comparisons=4,draws=10000,seed=73,adjustment='Bonferroni percentile intervals',individual_coverage=.9875,nominal_family_coverage=.95),
        confirmed_scoped_advantages=[k for k,v in primary.items() if v['adjusted_verdict']=='advantage'],
        confirmed_scoped_disadvantages=[k for k,v in primary.items() if v['adjusted_verdict']=='disadvantage'],
        confirmed_same_view_shared_advantage=[v for v in VIEWS if all(primary[g+'_'+v]['adjusted_verdict']=='advantage' for g in ('VT','HotpotQA'))],
        individual_tasks_and_non10_budgets_are_secondary=True)
