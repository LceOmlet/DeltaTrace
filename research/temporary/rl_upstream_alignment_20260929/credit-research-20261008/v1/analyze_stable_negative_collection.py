"""Join already measured references on the unchanged frozen development set.

Observed ranges are descriptive, not confidence intervals or tolerances.
No source is reselected, no coefficient is corrected, and no model is called.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(REPO)).replace('\\', '/'),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def describe(points):
    if not points:
        return dict(points=0, states=0, trajectories=0)
    flags = ('DT_tail_all_repeats', 'spurious_DT_tail_all_references',
             'missed_native_tail_all_references', 'opposite_sign_all_references',
             'probability_bound_violated_all_references')
    return dict(points=len(points),
        states=len({p['initial_state_sha256'] for p in points}),
        trajectories=len({p['traj_uid'] for p in points}),
        counts={k: sum(p[k] for p in points) for k in flags},
        distributions={key: quantiles([p[key] for p in points]) for key in (
            'DT_d', 'fresh_DT_d', 'FP32_head_d',
            'native_d_observed_range', 'distance_between_DT_and_reference_ranges',
            'head_projection_d_difference', 'DT_factual_score_drift')})


def main():
    started = time.perf_counter()
    evidence = [ref(HERE/'manifest.json'), ref(Path(__file__))]
    tasks = {}
    for task in ('textcraft', 'appworld'):
        layers = {}
        for rank in (0, 1):
            path = HERE/f'layer-{task}-observations/rank{rank}.json'
            evidence.append(ref(path))
            report = json.loads(path.read_bytes())
            assert report['phase'] == 'complete'
            for batch in report['batches']:
                for p in batch['points']:
                    key = (p['traj_uid'], p['packed_slot'])
                    assert key not in layers
                    row = batch['uids'].index(p['traj_uid'])
                    layers[key] = dict(p, DT_factual_logp=
                        batch['DT_detail']['per_sample'][row]['factual_target_logp'])
        path = REPO/f'experiments/rl/results_native_head_{task}_20261008.json'
        evidence.append(ref(path))
        heads = json.loads(path.read_bytes())
        cohorts = {}
        identities = set()
        for cohort, data in heads['cohorts'].items():
            points = []
            for p in data['points_with_identity']:
                key = (p['traj_uid'], p['packed_slot'])
                layer = layers[key]
                assert layer['token_id'] == p['token_id']
                assert layer['initial_state_sha256'] == p['initial_state_sha256']
                identities.add(key)
                # Includes separate native executions and a same-hidden FP32
                # output projection. This is not a full FP32 model reference.
                native = [p['old_native_d'], layer['native_single_d'],
                          p['native_d'], p['FP32_d']]
                predicted = [p['old_DT_d'], layer['fresh_DT_d']]
                factual = [old['factual_target_logp'] for old in layer['previous_comparisons']]
                factual += [layer['factual_target_logp'], layer['DT_factual_logp'],
                            p['scores']['native']['factual'], p['scores']['FP32']['factual']]
                nlo, nhi = min(native), max(native)
                dlo, dhi = min(predicted), max(predicted)
                points.append(dict(traj_uid=key[0], packed_slot=key[1],
                    token_id=p['token_id'], initial_state_sha256=p['initial_state_sha256'],
                    previously_examined=p['previously_examined'],
                    DT_d=p['old_DT_d'], fresh_DT_d=layer['fresh_DT_d'],
                    FP32_head_d=p['FP32_d'], native_d_values=native,
                    native_d_interval=[nlo,nhi], DT_d_interval=[dlo,dhi],
                    native_d_observed_range=nhi-nlo,
                    distance_between_DT_and_reference_ranges=max(0, nlo-dhi, dlo-nhi),
                    head_projection_d_difference=abs(p['native_d']-p['FP32_d']),
                    DT_factual_score_drift=layer['DT_factual_logp']-layer['factual_target_logp'],
                    DT_tail_all_repeats=dhi < -math.log(2),
                    spurious_DT_tail_all_references=dhi < -math.log(2) and nlo >= 0,
                    missed_native_tail_all_references=dlo >= -math.log(2) and nhi < -math.log(2),
                    opposite_sign_all_references=(dhi < 0 <= nlo or nhi < 0 <= dlo),
                    # p_deleted<=1 implies d>=log(p_factual). Check all
                    # observed scores; report a violation, never clamp it.
                    probability_bound_violated_all_references=dhi < min(factual),
                    DT_ratio_bin=stratum(p['old_DT_d']),
                    FP32_head_ratio_bin=stratum(p['FP32_d']),
                    DT_A_over_r=-math.expm1(-p['old_DT_d']),
                    FP32_head_A_over_r=-math.expm1(-p['FP32_d']),
                    factual_logp_observed_interval=[min(factual),max(factual)]))
            assert len(points) == (128 if cohort == 'uniform' else 37)
            groups = {}
            for p in points:
                cell = p['DT_ratio_bin']+':'+p['FP32_head_ratio_bin']
                groups.setdefault(cell, []).append(p)
            cohorts[cohort] = dict(denominator=len(points),
                counts={k: sum(p[k] for p in points) for k in (
                    'DT_tail_all_repeats','spurious_DT_tail_all_references',
                    'missed_native_tail_all_references','opposite_sign_all_references',
                    'probability_bound_violated_all_references')},
                cells={k: dict(all=describe(v), by_state={s:describe([
                    p for p in v if p['initial_state_sha256']==s]) for s in sorted({
                    p['initial_state_sha256'] for p in v})},
                    by_exposure={str(e):describe([p for p in v if p['previously_examined']==e])
                                 for e in (False,True)}) for k,v in sorted(groups.items())},
                points_with_identity=points)
        assert len(identities) == 165 and identities == set(layers)
        tasks[task] = dict(unique_points=165, cohorts=cohorts)
    result = dict(scope=__doc__, observed_unix=time.time(), inputs=evidence, tasks=tasks,
        interpretation=[
            'Uniform source sample and predicted-tail census retain their own denominators; no pooled raw tail/body moments.',
            'Observed-range separation is descriptive robustness across saved executions, not an official acceptance assertion or statistical confidence bound.',
            'FP32 head uses the same BF16 hidden states and weights; hidden-state and whole-model precision error remain distinct.',
            'The necessary probability bound detects impossible inferred deletion probabilities under the corresponding factual score, not the responsible internal operator.',
            'Layer residuals contract the fixed joint DT coefficients with a different native single-deletion pair. Their factual endpoint mismatch prevents interpreting them as pure local finite-rule error.',
            'No component cause, production repair or task performance improvement follows from this join alone.'
        ], operations=dict(model=0,DT=0,GPU=0,backward=0,optimizer=0,
                           production_modified=False),
        elapsed_seconds=time.perf_counter()-started)
    output = REPO/'experiments/rl/results_stable_negative_credit_20261008.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),tasks={task:{cohort:{
        'denominator':data['denominator'],**data['counts']} for cohort,data in value['cohorts'].items()}
        for task,value in tasks.items()},operations=result['operations'],
        elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
