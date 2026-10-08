"""Summarize a passive paired native-head comparison, without new tolerances."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_native_reference_drift import describe as describe_drift
from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(REPO)).replace('\\', '/'),
        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def describe(points):
    return dict(points=len(points), trajectories=len({p['traj_uid'] for p in points}),
        states=len({p['initial_state_sha256'] for p in points}),
        abs_native_to_FP32_d=quantiles([abs(p['head_projection_d_difference']) for p in points]),
        abs_same_FP32_logits_rounding_d=quantiles([abs(p['isolated_output_rounding_d']) for p in points]),
        abs_native_to_rounded_FP32_d=quantiles([abs(p['native_to_rounded_FP32_d']) for p in points]),
        native_FP32_sign_crossings=sum((p['native_d'] < 0) != (p['FP32_d'] < 0) for p in points),
        old_missed_tail=sum(p['old_native_d'] < -math.log(2) and p['old_DT_d'] >= -math.log(2) for p in points),
        old_missed_tail_still_native_tail=sum(p['old_native_d'] < -math.log(2) and p['old_DT_d'] >= -math.log(2)
            and p['native_d'] < -math.log(2) for p in points),
        old_missed_tail_still_FP32_tail=sum(p['old_native_d'] < -math.log(2) and p['old_DT_d'] >= -math.log(2)
            and p['FP32_d'] < -math.log(2) for p in points),
        old_predicted_tail_FP32_native_nonnegative=sum(p['old_DT_d'] < -math.log(2) and p['FP32_d'] >= 0 for p in points))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('textcraft', 'appworld'), required=True)
    args = parser.parse_args()
    directory = HERE/f'native-head-{args.task}-final'
    paths = [directory/f'rank{rank}.json' for rank in (0, 1)]
    reports = [json.loads(p.read_bytes()) for p in paths]
    assert all(r['phase'] == 'complete' for r in reports), 'Do not present a partial collection as complete'
    cohorts = dict(uniform=[], predicted_tail_census=[])
    keys = set()
    for report in reports:
        assert all(report['operations'][k] == 0 for k in ('DT', 'optimizer', 'backward', 'rollout', 'checkpoint_restore', 'production_patches'))
        for batch in report['batches']:
            for point in batch['points']:
                key = (point['traj_uid'], point['packed_slot'])
                assert key not in keys
                keys.add(key)
                scores = point['scores']
                assert all(math.isfinite(v) for s in scores.values() for v in s.values())
                for previous in point['previous_comparisons']:
                    native, reference, rounded = (scores[k]['d'] for k in ('native', 'FP32', 'rounded_FP32'))
                    cohorts[previous['cohort']].append(dict(
                        traj_uid=point['traj_uid'], packed_slot=point['packed_slot'], token_id=point['token_id'],
                        initial_state_sha256=point['initial_state_sha256'], previously_examined=point['previously_examined'],
                        old_DT_d=previous.get('saved_d', previous.get('d')), old_native_d=previous['native_single_d'],
                        native_d=native, FP32_d=reference, rounded_FP32_d=rounded,
                        head_projection_d_difference=native-reference,
                        isolated_output_rounding_d=rounded-reference,
                        native_to_rounded_FP32_d=native-rounded,
                        factual_score_drift=scores['native']['factual']-previous['factual_target_logp'],
                        deleted_score_drift=scores['native']['deleted']-previous['deleted_target_logp'],
                        fresh_native_d=native, scores=scores))
    assert len(keys) == 165 and len(cohorts['uniform']) == 128 and len(cohorts['predicted_tail_census']) == 37
    data = {}
    for cohort, points in cohorts.items():
        grouped = {}
        for point in points:
            cell = ':'.join(stratum(point[k]) for k in ('old_DT_d', 'old_native_d', 'FP32_d'))
            grouped.setdefault(cell, []).append(point)
        data[cohort] = dict(summary=describe(points), native_repeat_drift=describe_drift(points),
            old_DT_old_native_FP32_native_cells={k: describe(v) for k, v in sorted(grouped.items())},
            by_state={k: describe([p for p in points if p['initial_state_sha256'] == k])
                for k in sorted({p['initial_state_sha256'] for p in points})},
            points_with_identity=points)
    result = dict(scope=__doc__, task=args.task, inputs=[ref(p) for p in paths],
        script=ref(Path(__file__)), reference=reports[0]['reference'], cohorts=data,
        phases=[dict(rank=r['rank'], elapsed_seconds=r['elapsed_seconds'],
            native_forward_calls=r['operations']['native_forward'], peak_allocated=r['peak_allocated'],
            native_forward_seconds=sum(p['seconds'] for b in r['batches'] for p in b['native_phases']),
            head_reference_seconds=sum(p['head_reference_seconds'] for b in r['batches'] for p in b['native_phases'])) for r in reports],
        DT=0, optimizer=0, rollout=0, production_modified=False,
        interpretation=[
            'FP32 is a same-hidden/same-weight projection reference, not a full FP32 model or world counterfactual.',
            'Rounding the same FP32 logits isolates storage rounding. Native BF16 versus FP32 also includes GEMM differences.',
            'No official FA/FLA or PPO acceptance assertion is applied to this head diagnostic.',
            'These measurements bound the native reference-side head difference; they do not measure the effect of changing the DT seed or justify correcting token credit.'
        ])
    output = REPO/f'experiments/rl/results_native_head_{args.task}_20261008.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(task=args.task, output=str(output),
        summaries={k: v['summary'] for k, v in data.items()}, phases=result['phases'])))


if __name__ == '__main__':
    main()
