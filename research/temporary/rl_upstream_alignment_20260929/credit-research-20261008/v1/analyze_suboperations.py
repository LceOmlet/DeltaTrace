"""Localize existing errors; never convert passive residuals into credit.

Tasks, source-uniform and predicted-tail cohorts, prediction/reference ratio
cross-cells and exposure remain separate. Signed conditional quantiles are
descriptive finite-sample measurements, not population moments or cause shares.
"""
import argparse
import json
from pathlib import Path

from analyze_layer_collection import ref
from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent


def summarize(points):
    if not points:
        return dict(points=0, states=0, status='Empty cell; no values imputed')
    result = dict(points=len(points), states=len({p['initial_state_sha256'] for p in points}),
                  trajectories=len({p['traj_uid'] for p in points}), operations={})
    for mode in ('native','matched'):
        operations = {}
        for point in points:
            for location, data in point['suboperations'].items():
                for name, value in data[mode+'_terms'].items():
                    operations.setdefault(location+'/'+name, []).append(value)
        result['operations'][mode] = {k:dict(signed=quantiles(v), absolute=quantiles([abs(x) for x in v]),
            negative=sum(x<0 for x in v), positive=sum(x>0 for x in v)) for k,v in operations.items()}
        result[mode+'_closure_roundoff'] = quantiles([
            data[mode+'_telescoping_roundoff'] for p in points for data in p['suboperations'].values()])
    # Frequencies are bounded and state-balanced. They identify which measured
    # operation is largest, not how much of final error it caused.
    leaders = {}
    for p in points:
        for location, data in p['suboperations'].items():
            winner = max(data['matched_terms'], key=lambda k:abs(data['matched_terms'][k]))
            leaders.setdefault(location,{}).setdefault(p['initial_state_sha256'],{}).setdefault(p['traj_uid'],[]).append(winner)
    result['state_equal_largest_abs_term_frequency'] = {}
    for location, states in leaders.items():
        names = sorted({w for trajectories in states.values() for values in trajectories.values() for w in values})
        result['state_equal_largest_abs_term_frequency'][location] = {
            name:sum(sum(sum(w==name for w in values)/len(values) for values in trajectories.values())/len(trajectories)
                     for trajectories in states.values())/len(states) for name in names}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=('textcraft','appworld'))
    parser.add_argument('--revision',choices=('v1','v2'),default='v2')
    args = parser.parse_args()
    task = args.task
    tag = f'layer-suboperations-{task}'+('-'+args.revision if args.revision!='v1' else '')
    folder = HERE/(tag+'-observations')
    paths = [folder/f'rank{r}.json' for r in (0,1)]
    ranks = [json.loads(p.read_bytes()) for p in paths]
    assert all(r['phase']=='complete' for r in ranks)
    points = [p for r in ranks for b in r['batches'] for p in b['points']]
    baseline_paths = [HERE/f'layer-factual-controls-{task}-observations/rank{r}.json' for r in (0,1)]
    baseline = {(p['traj_uid'],p['packed_slot']):p for f in baseline_paths
                for b in json.loads(f.read_bytes())['batches'] for p in b['points']}
    assert len(points)==165 and len(baseline)==165
    assert {(p['traj_uid'],p['packed_slot']) for p in points} == set(baseline)
    stable_path = HERE.parents[4]/'experiments/rl/results_stable_negative_credit_20261008.json'
    stable = json.loads(stable_path.read_bytes())['tasks'][task]['cohorts']
    flags = {(p['traj_uid'],p['packed_slot']):p for c in stable.values() for p in c['points_with_identity']}
    comparisons = []
    for p in points:
        old = baseline[p['traj_uid'],p['packed_slot']]
        comparisons.append(dict(traj_uid=p['traj_uid'], packed_slot=p['packed_slot'],
            DT_difference=p['fresh_DT_d']-old['fresh_DT_d'], native_difference=p['native_single_d']-old['native_single_d'],
            maximum_boundary_difference=max(abs(p['boundaries'][i]['value']-old['boundaries'][i]['value']) for i in p['boundaries'])))
    cohorts = {}
    for cohort in ('uniform','predicted_tail_census'):
        rows = [p for p in points if any(c['cohort']==cohort for c in p['previous_comparisons'])]
        cells = {}
        for p in rows:
            c = next(c for c in p['previous_comparisons'] if c['cohort']==cohort)
            cell = stratum(c.get('saved_d',c.get('d')))+':'+stratum(c['native_single_d'])
            cells.setdefault(cell,[]).append(p)
        cohorts[cohort] = dict(points=len(rows), crossed_ratio_cells={
            cell:dict(all=summarize(items), by_exposure={str(seen):summarize([p for p in items if p['previously_examined']==seen])
                         for seen in (False,True)}) for cell,items in cells.items()},
            robust_error_subsets={flag:dict(
                points=sum(flags[p['traj_uid'],p['packed_slot']][flag] for p in rows),
                states=len({p['initial_state_sha256'] for p in rows if flags[p['traj_uid'],p['packed_slot']][flag]}),
                crossed_ratio_cells={cell:summarize([p for p in items if flags[p['traj_uid'],p['packed_slot']][flag]])
                    for cell,items in cells.items()})
                for flag in ('spurious_DT_tail_all_references','missed_native_tail_all_references','probability_bound_violated_all_references')})
    result = dict(scope=__doc__,task=task,complete=True,unique_points=len(points),
        inputs=[ref(p) for p in paths+baseline_paths+[stable_path]], cohorts=cohorts,
        unchanged_output_comparisons=comparisons,
        operations=[r['operations'] for r in ranks], elapsed_seconds=[r['elapsed_seconds'] for r in ranks],
        maximum_observed_retained_host_bank_bytes=max(b['retained_bank_bytes']+b['retained_suboperation_bytes'] for r in ranks for b in r['batches']),
        diagnostic_readout_seconds=[sum(b['additional_suboperation_readout_seconds'] for b in r['batches']) for r in ranks],
        official_tolerance_test=False,candidate=False,production_modified=False,
        interpretation='Exact owner output differences are repeatability observations. Suboperation residuals apply jointly computed coefficients to individual deletions and include finite-background approximation. Neither a large term nor its frequency alone proves a kernel bug. Head seed recomputation differences and native BF16 residual-add effects remain explicit. No residual is subtracted from credit, and this is not the author RISE/MAS acceptance test.')
    output = HERE/(tag+'-analysis.json')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),task=task,points=len(points),
        identical_DT=sum(c['DT_difference']==0 for c in comparisons),
        max_native_difference=max(abs(c['native_difference']) for c in comparisons),
        robust_error_subsets={cohort:{flag:dict(points=v['points'],states=v['states'])
            for flag,v in data['robust_error_subsets'].items()} for cohort,data in cohorts.items()})))


if __name__=='__main__':
    main()
