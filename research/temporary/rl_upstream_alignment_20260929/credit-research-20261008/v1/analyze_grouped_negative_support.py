"""Grouped, exploratory support for negative EOS-DT credit on frozen data.

Reuse all recorded references, without new model calls or selecting examples.
Cluster bootstrap intervals concern state-to-state variation of bounded rates,
conditional on this development collection. They are not numerical tolerances,
population guarantees, or a replacement for author cumulative deletion tests.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time

from summarize_author_collection import quantiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SEED = 20261009
REPLICATES = 10000


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def flags(point):
    lo, hi = point['native_d_interval']
    return dict(native_negative_all_saved_references=hi < 0,
                native_nonnegative_all_saved_references=lo >= 0,
                native_sign_unresolved=lo < 0 <= hi,
                native_large_negative_all_saved_references=hi < -math.log(2),
                native_large_negative_excluded_by_all_saved_references=lo >= -math.log(2),
                DT_large_negative_all_saved_references=point['DT_tail_all_repeats'],
                robust_spurious_negative_tail=point['spurious_DT_tail_all_references'],
                robust_missed_negative_tail=point['missed_native_tail_all_references'],
                probability_bound_violated_all_saved_references=point['probability_bound_violated_all_references'])


def summarize(points):
    groups = {}
    for p in points:
        groups.setdefault(p['initial_state_sha256'], {}).setdefault(p['traj_uid'], []).append(p)
    result = dict(points=len(points), states=len(groups),
                  trajectories=len({p['traj_uid'] for p in points}))
    if not points:
        return dict(**result, missing='Empty cell, no zero imputation')
    result['counts'] = {name:sum(flags(p)[name] for p in points) for name in flags(points[0])}
    result['bounded_state_equal_frequencies'] = {}
    for name in flags(points[0]):
        state_rates = {state:sum(sum(flags(p)[name] for p in rows)/len(rows)
                            for rows in trajectories.values())/len(trajectories)
                       for state,trajectories in groups.items()}
        values = list(state_rates.values())
        rng = random.Random(SEED)
        draws = sorted(sum(rng.choices(values,k=len(values)))/len(values)
                       for _ in range(REPLICATES)) if len(values)>1 else None
        result['bounded_state_equal_frequencies'][name] = dict(
            estimate=sum(values)/len(values), by_state=state_rates,
            exploratory_state_cluster_bootstrap_95_percentile_interval=
                [draws[int(.025*REPLICATES)],draws[int(.975*REPLICATES)-1]] if draws else None)
    return result


def main():
    start=time.perf_counter()
    source=REPO/'experiments/rl/results_stable_negative_credit_20261008.json'
    recorded=json.loads(source.read_bytes())
    latest_paths=[HERE/f'attention-pv-textcraft-v2/rank{r}.json' for r in (0,1)]
    latest={(p['traj_uid'],p['packed_slot']):p for path in latest_paths
            for batch in json.loads(path.read_bytes())['batches'] for p in batch['points']}
    tasks={}
    for task,original in recorded['tasks'].items():
        cohorts={}
        for cohort,data in original['cohorts'].items():
            points=[dict(p) for p in data['points_with_identity']]
            if task=='textcraft':
                for p in points:
                    current=latest[p['traj_uid'],p['packed_slot']]
                    assert current['token_id']==p['token_id']
                    p['native_d_interval']=[min(p['native_d_interval'][0],current['native_single_d']),
                                            max(p['native_d_interval'][1],current['native_single_d'])]
                    p['DT_d_interval']=[min(p['DT_d_interval'][0],current['fresh_DT_d']),
                                        max(p['DT_d_interval'][1],current['fresh_DT_d'])]
                    p['DT_tail_all_repeats']=p['DT_d_interval'][1]<-math.log(2)
                    p['spurious_DT_tail_all_references']=p['DT_tail_all_repeats'] and p['native_d_interval'][0]>=0
                    p['missed_native_tail_all_references']=p['DT_d_interval'][0]>=-math.log(2) and p['native_d_interval'][1]<-math.log(2)
                    p['probability_bound_violated_all_references']=p['DT_d_interval'][1]<min(
                        p['factual_logp_observed_interval'][0],current['factual_target_logp'],current['DT_factual_target_logp'])
            cells={}
            for p in points:
                cells.setdefault(p['DT_ratio_bin']+':'+p['FP32_head_ratio_bin'],[]).append(p)
            cohorts[cohort]=dict(
                all=summarize(points),
                by_exposure={str(e):summarize([p for p in points if p['previously_examined']==e])
                             for e in (False,True)},
                predicted_native_cross_cells={cell:dict(
                    all=summarize(rows),
                    conditional_finite_sample_quantiles={
                        name:quantiles([p[name] for p in rows]) for name in
                        ('DT_d','FP32_head_d','DT_A_over_r','FP32_head_A_over_r',
                         'distance_between_DT_and_reference_ranges')},
                    by_exposure={str(e):summarize([p for p in rows if p['previously_examined']==e])
                                 for e in (False,True)}) for cell,rows in sorted(cells.items())},
                points_with_identity=[dict(traj_uid=p['traj_uid'],packed_slot=p['packed_slot'],
                    token_id=p['token_id'],initial_state_sha256=p['initial_state_sha256'],
                    previously_examined=p['previously_examined'],DT_d_interval=p['DT_d_interval'],
                    native_d_interval=p['native_d_interval'],
                    DT_A_over_r=p['DT_A_over_r'],FP32_head_A_over_r=p['FP32_head_A_over_r'],
                    DT_ratio_bin=p['DT_ratio_bin'],FP32_head_ratio_bin=p['FP32_head_ratio_bin'],
                    **flags(p)) for p in points])
        tasks[task]=dict(unique_points=original['unique_points'],cohorts=cohorts)
    result=dict(scope=__doc__,observed_unix=time.time(),sources=[ref(source),ref(Path(__file__)),*[ref(p) for p in latest_paths]],
        statistical_method=dict(aggregation='Source rate per trajectory, trajectory mean per initial state, state equal mean',
            cluster_unit='Complete initial state, preserving its trajectories and source positions',
            bootstrap_replicates=REPLICATES,seed=SEED,
            interval='Exploratory percentile 95% interval, assuming these states represent exchangeable draws; assumption not established by this selected development collection',
            fewer_than_two_states='No interval; repeated tokens do not replace independent states',
            reference_range='Observed execution/precision range, not a confidence interval',
            bins='Existing PLAN c=exp(-d) bins; no learned threshold or new tolerance'),
        tasks=tasks,
        interpretation=[
            'Uniform source sample and predicted-tail census have separate denominators; no pooled raw tail/body moments.',
            'Negative credit for positive reward means deletion raises the probability of the same actual executed target. This is permitted, but its magnitude needs counterfactual support.',
            'All-reference nonnegative credit contradicts a predicted negative tail on that frozen point, and indicates potential wrong-direction policy pressure; this is not measured gradient share.',
            'Group average agreement cannot certify individual tail tokens or erase an unsupported tail. Native sign crossing zero remains unresolved.',
            'Native single deletion is a supplemental diagnostic under the agreed model/world idealization. It does not replace original author RISE/MAS or official actual-dtype FA/FLA checks.',
            'The collection originates in saved nonzero-reward requests, not an unbiased sample of all task trajectories or deployed training effects.',
            'TextCraft robustness flags include the latest completed PV execution. Original predicted/native FP32-head bins and their descriptive quantiles remain fixed, not reselected after the extra execution.',
            'No production change, credit clipping/scaling, additional reference sampling, or numerical repair is made.'
        ],operations=dict(model=0,DT=0,FA=0,GPU=0,optimizer=0,production_modified=False),
        elapsed_seconds=time.perf_counter()-start)
    output=REPO/'experiments/rl/results_grouped_negative_support_20261009.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=ref(output),elapsed_seconds=result['elapsed_seconds'],summary={
        t:{k:v for k,v in c['cohorts']['predicted_tail_census']['all'].items()
           if k!='bounded_state_equal_frequencies'} for t,c in tasks.items()})))


if __name__=='__main__':
    main()
