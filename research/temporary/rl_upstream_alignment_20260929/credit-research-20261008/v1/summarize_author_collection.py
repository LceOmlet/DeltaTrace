"""Keep original bounded author metrics, uniform diagnostics and tail census separate.

Single-EOS native scores are measured model endpoints, not world-simulator
truth. All quantiles/counts below describe only the declared frozen sample.
There is no pooled raw advantage/error mean, tail correction or training gate.
"""
from bisect import bisect_left
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

from score_summary import summarize

HERE=Path(__file__).resolve().parent
NAMES=('ratio_le_1','ratio_1_to_2','ratio_2_to_10','ratio_10_to_100','ratio_gt_100')


def stratum(d):
    return NAMES[bisect_left((0.,math.log(2),math.log(10),math.log(100)),-d)] if d<0 else NAMES[0]


def quantiles(values):
    values=sorted(values)
    if not values:return None
    def at(p):
        location=(len(values)-1)*p;left=math.floor(location);right=math.ceil(location)
        return values[left]+(values[right]-values[left])*(location-left)
    return {'min':values[0],'p25':at(.25),'median':at(.5),'p75':at(.75),'p95':at(.95),'max':values[-1]}


def diagnostics(points):
    groups={name:[] for name in NAMES}
    nonfinite=[]
    for p in points:
        d=p.get('saved_d',p.get('d'));native=p['native_single_d']
        if not math.isfinite(d) or not math.isfinite(native):
            nonfinite.append(p);continue
        groups[stratum(d)].append(dict(p,saved_d=d,native_stratum=stratum(native)))
    output={}
    for name,items in groups.items():
        transitions={n:sum(p['native_stratum']==n for p in items) for n in NAMES}
        by_state={}
        for p in items:by_state.setdefault(p['initial_state_sha256'],[]).append(p)
        states=[]
        for state,rows in by_state.items():
            states.append({'initial_state_sha256':state,'points':len(rows),
                'sign_crossings':sum((p['saved_d']<0)!=(p['native_single_d']<0) for p in rows),
                'abs_d_error':quantiles([abs(p['saved_d']-p['native_single_d']) for p in rows])})
        normalized_errors=[];nonfinite_transforms=[]
        for p in items:
            try:
                a=-math.expm1(-p['saved_d']);ref=-math.expm1(-p['native_single_d'])
                error=abs(a-ref)
                if math.isfinite(error):normalized_errors.append(error)
                else:nonfinite_transforms.append([p['traj_uid'],p['packed_slot']])
            except OverflowError:nonfinite_transforms.append([p['traj_uid'],p['packed_slot']])
        output[name]={'points':len(items),'trajectories':len({p['traj_uid'] for p in items}),
            'initial_states':len(by_state),'native_stratum_transitions':transitions,
            'sign_crossings':sum((p['saved_d']<0)!=(p['native_single_d']<0) for p in items),
            'abs_d_error':quantiles([abs(p['saved_d']-p['native_single_d']) for p in items]),
            'abs_A_over_r_error':quantiles(normalized_errors),'nonfinite_transforms':nonfinite_transforms,
            'state_groups':states}
    return {'strata':output,'nonfinite_points':nonfinite,
        'sign_scope':'Descriptive crossing of zero, not an official numerical-tolerance failure.',
        'aggregation_scope':'Conditional empirical quantiles/counts per DT-ratio stratum. No cross-stratum raw mean or population-moment claim.'}


if __name__=='__main__':
    manifest=json.loads((HERE/'manifest.json').read_bytes())
    strata=json.loads((HERE/'credit-strata.json').read_bytes())
    stage=deepcopy(manifest)
    for task in stage['tasks'].values():
        for group in task['groups']:
            if group['split']=='development':group['trajectory_uids']=group['first_stage_uids']
    metrics=[];uniform={t:[] for t in manifest['tasks']};tails={t:[] for t in manifest['tasks']};bindings=[];phases={}
    identity={u:g for t in manifest['tasks'].values() for g in t['groups'] for u in g['trajectory_uids']}
    for rank in (0,1):
        path=HERE/'author-collection-observations'/f'rank{rank}.json'
        if not path.exists():continue
        raw=path.read_bytes();value=json.loads(raw)
        bindings.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
        phases[str(rank)]={'phase':value['phase'],'native_forward_calls':value['native_forward_calls'],
            'seconds':value.get('elapsed_seconds'),'peak_allocated':value.get('peak_allocated')}
        for batch in value['batches']:
            for row in batch['trajectories']:
                if 'signed_RISE' not in row['views']:continue
                metrics.append({'task':batch['task'],'traj_uid':row['traj_uid'],
                    'metrics':{'rise':row['views']['signed_RISE']['author_return'][0],
                               'mas':row['views']['positive_MAS']['author_return'][1]}})
                for q in row['uniform_deletions']:
                    uniform[batch['task']].append(dict(q,traj_uid=row['traj_uid'],
                        initial_state_sha256=row['initial_state_sha256']))
        for point in value.get('tail_results',[]):
            task=next(t for t,s in strata['tasks'].items() if point['traj_uid'] in s['rows'])
            tails[task].append(point)
    result={'scope':__doc__,'sources':bindings,'phases':phases,
        'author_metrics_first_stage':summarize(stage,metrics,'development'),
        'uniform_sample':{t:{'expected_points':128,'measured_points':len(p),'results':diagnostics(p)} for t,p in uniform.items()},
        'observed_DT_tail_census':{t:{'expected_points':len(strata['tasks'][t]['complete_observed_ratio_gt_2_tail']),
            'measured_points':len(p),'results':diagnostics(p)} for t,p in tails.items()},
        'policy_gradient_influence':'Unmeasured by this forward-only diagnostic. Coefficient error is not a gradient share.',
        'historical_degradation_cause':'Not established by this current first-rollout corpus.'}
    for name,sets in (('uniform_sample',uniform),('observed_DT_tail_census',tails)):
        for task,points in sets.items():
            result[name][task]['by_exposure']={str(exposed):diagnostics([
                p for p in points if identity[p['traj_uid']]['previously_examined']==exposed]) for exposed in (False,True)}
    result['author_metrics_by_exposure']={}
    for exposed in (False,True):
        subset=deepcopy(stage)
        for task in subset['tasks'].values():
            task['groups']=[g for g in task['groups'] if g['previously_examined']==exposed]
        rows=[r for r in metrics if identity[r['traj_uid']]['previously_examined']==exposed]
        result['author_metrics_by_exposure'][str(exposed)]=summarize(subset,rows,'development')
    path=HERE/'author-collection-summary.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    metric_brief={t:{m:{k:v for k,v in entry.items() if k not in ('missing_uids','nonfinite_uids')}
        for m,entry in spec.items()} for t,spec in result['author_metrics_first_stage'].items()}
    print(json.dumps({'path':str(path),'phases':phases,'author_metrics':metric_brief,
        'uniform_measured':{t:len(p) for t,p in uniform.items()},'tail_measured':{t:len(p) for t,p in tails.items()}},ensure_ascii=False))
