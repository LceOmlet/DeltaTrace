"""Aggregate paired original-author metrics; preserve tail cohorts and missingness."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def quantiles(values):
    total = len(values)
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return dict(n=0, nonfinite=total)
    return dict(n=len(values), nonfinite=total-len(values), min=values[0], median=statistics.median(values), max=values[-1])


def advantage_error(q, label):
    try:
        return abs(math.expm1(-q['fresh_'+label+'_d'])-math.expm1(-q['fresh_native_single_d']))
    except OverflowError:
        return math.inf


def ratio_bin(d):
    if d >= 0:
        return '[0,1]'
    for upper in (2,10,100):
        if -d <= math.log(upper):
            return '(1,2]' if upper == 2 else '(2,10]' if upper == 10 else '(10,100]'
    return '(100,inf)'


def analyze(plan, records):
    task = next(iter(plan['tasks']))
    data = plan['tasks'][task]
    expected = {uid for b in data['batches'] if b['primary'] for uid in b['uids']}
    entries = {entry['traj_uid']:entry for entry in data['entries']}
    rows = {}
    queries = {}
    for record in records:
        for batch in record['batches']:
            for row in batch['trajectories']:
                uid = row['traj_uid']
                for label, views in row['views'].items():
                    for view, result in views.items():
                        key = (uid,label,view)
                        assert key not in rows or rows[key] == result
                        rows[key] = result
                for query in row['single_deletions']:
                    key = (uid,query['packed_slot'])
                    assert key not in queries or queries[key] == query
                    queries[key] = query
    metrics = {}
    for view, index in [('signed_RISE',0),('positive_MAS',1)]:
        paired = []
        missing, nonfinite = [], []
        for uid in sorted(expected):
            before, after = rows.get((uid,'original',view)), rows.get((uid,'conditional',view))
            if before is None or after is None:
                missing.append(uid)
                continue
            before, after = before['author_return'][index], after['author_return'][index]
            if not math.isfinite(before) or not math.isfinite(after):
                nonfinite.append(uid)
                continue
            paired.append(dict(traj_uid=uid,state=entries[uid]['initial_state_sha256'],
                original=before,conditional=after,delta=after-before))
        states = {}
        for row in paired:
            states.setdefault(row['state'],[]).append(row)
        state_means = {state:{name:statistics.mean(row[name] for row in values)
            for name in ('original','conditional','delta')} for state,values in states.items()}
        deltas = [value['delta'] for value in state_means.values()]
        metrics[view] = dict(expected=len(expected),paired_finite=len(paired),missing_uids=missing,
            nonfinite_uids=nonfinite,complete=not missing and not nonfinite,
            available_equal_state_means={name:statistics.mean(v[name] for v in state_means.values())
                for name in ('original','conditional','delta')} if state_means else None,
            state_delta_standard_error=statistics.stdev(deltas)/math.sqrt(len(deltas)) if len(deltas)>1 else None,
            state_means=state_means,paired_trajectories=paired)
    cohorts = {}
    for cohort in ('uniform','predicted_tail_census'):
        points = [dict(q,traj_uid=uid,state=entries[uid]['initial_state_sha256'])
            for (uid,_),q in queries.items() if cohort in q['cohorts']]
        cells = {}
        for q in points:
            cell = (ratio_bin(q['fresh_original_d']),ratio_bin(q['fresh_native_single_d']))
            cells.setdefault(cell,[]).append(q)
        cohorts[cohort] = dict(measured_points=len(points),
            expected_points=sum(cohort in q['cohorts'] for e in data['entries'] for q in e['queries']),
            cells=[dict(original_ratio_bin=cell[0],native_ratio_bin=cell[1],points=len(values),
                initial_states=len({q['state'] for q in values}),
                original_abs_d_error=quantiles([abs(q['fresh_original_d']-q['fresh_native_single_d']) for q in values]),
                conditional_abs_d_error=quantiles([abs(q['fresh_conditional_d']-q['fresh_native_single_d']) for q in values]),
                original_abs_A_over_r_error=quantiles([advantage_error(q,'original') for q in values]),
                conditional_abs_A_over_r_error=quantiles([advantage_error(q,'conditional') for q in values]),
                original_sign_crossings=sum((q['fresh_original_d']<0)!=(q['fresh_native_single_d']<0) for q in values),
                conditional_sign_crossings=sum((q['fresh_conditional_d']<0)!=(q['fresh_native_single_d']<0) for q in values))
                for cell,values in cells.items()], points_with_identity=points)
    return dict(task=task,complete=all(r['phase']=='complete' for r in records)
        and len(records)==2,primary_metrics=metrics,cohorts=cohorts,
        metric_owner=dict(plan['metric_owner'],direction='lower_is_better',
            owner_return='auc(normalized_model_response), auc(corrected_scores), auc(normalized_model_response + alignment_penalty)'),
        interpretation='Original-author per-trajectory scores, then equal initial-state means. Tail and uniform denominators separate. Only original/native joint cells aggregate conditional magnitude. No pooled tail/bulk moment or training-effect claim.',
        candidate_deployed=False,selection_or_tuning_performed=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--records',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(json.loads(args.inputs.read_bytes()),[json.loads(p.read_bytes()) for p in args.records])
    result['evidence']=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [args.inputs,*args.records]]
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(task=result['task'],complete=result['complete'],metrics={k:{n:v for n,v in item.items()
        if n not in ('paired_trajectories','state_means','missing_uids','nonfinite_uids')} for k,item in result['primary_metrics'].items()}),ensure_ascii=False))
