"""Measure existing factual score drift without treating it as a causal error share."""
import hashlib
import json
from pathlib import Path

from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(REPO)).replace('\\', '/'),
                sha256=hashlib.sha256(raw).hexdigest())


def describe(points):
    return dict(points=len(points),
        trajectories=len({p['traj_uid'] for p in points}),
        states=len({p['initial_state_sha256'] for p in points}),
        absolute_factual_score_drift=quantiles([abs(p['factual_drift']) for p in points]),
        absolute_d_error=quantiles([abs(p['d_error']) for p in points]),
        factual_drift_smaller_than_d_error=sum(abs(p['factual_drift']) < abs(p['d_error']) for p in points))


def main():
    tasks = {}
    for task in ('textcraft', 'appworld'):
        paths = [HERE/f'layer-{task}-observations/rank{rank}.json' for rank in (0, 1)]
        cohorts = {'uniform': [], 'predicted_tail_census': []}
        for path in paths:
            report = json.loads(path.read_bytes())
            assert report['phase'] == 'complete'
            for batch in report['batches']:
                for point in batch['points']:
                    row = batch['uids'].index(point['traj_uid'])
                    factual = batch['DT_detail']['per_sample'][row]['factual_target_logp']
                    for previous in point['previous_comparisons']:
                        cell = stratum(previous.get('saved_d', previous.get('d'))) + ':' + stratum(previous['native_single_d'])
                        cohorts[previous['cohort']].append(dict(
                            traj_uid=point['traj_uid'], packed_slot=point['packed_slot'],
                            initial_state_sha256=point['initial_state_sha256'],
                            previously_examined=point['previously_examined'],
                            crossed_ratio_cell=cell,
                            DT_factual_target_logp=factual,
                            native_factual_target_logp=point['factual_target_logp'],
                            factual_drift=factual-point['factual_target_logp'],
                            d_error=point['fresh_DT_d']-point['native_single_d']))
        cells = {}
        for cohort, points in cohorts.items():
            grouped = {}
            for point in points:
                grouped.setdefault(point['crossed_ratio_cell'], []).append(point)
            cells[cohort] = dict(points=len(points), cells={cell: dict(
                all=describe(items),
                by_state={state: describe([p for p in items if p['initial_state_sha256'] == state])
                          for state in sorted({p['initial_state_sha256'] for p in items})},
                points_with_identity=items) for cell, items in sorted(grouped.items())})
        tasks[task] = dict(inputs=[ref(p) for p in paths], cohorts=cells)
    result = dict(scope=__doc__, source=ref(Path(__file__)), tasks=tasks,
        comparison_context='Same frozen collection points; DT cached-prefix factual score versus native full forward factual score. Both use the scoped original FA/FLA precision and target readers in the original collection observer.',
        interpretation=[
            'These are observed score differences, not an official FA/FLA numerical tolerance test.',
            'A factual scalar score offset is not the effect of changing cached-prefix replay on all finite coefficients.',
            'This comparison cannot identify a kernel as the cause, rule out precision effects, or justify subtracting a residual.',
            'Keep source error and measured endpoint drift separate; do not call either a complete repair or a training-degradation explanation.'
        ], model_calls=0, DT_calls=0, updates=0, production_modified=False)
    output = REPO/'experiments/rl/results_extreme_attribution_endpoint_drift_20261008.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), task_points={task: {c: v['points'] for c, v in data['cohorts'].items()}
        for task, data in tasks.items()}, model_calls=0)))


if __name__ == '__main__':
    main()
