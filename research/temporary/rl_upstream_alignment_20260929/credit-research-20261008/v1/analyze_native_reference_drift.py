"""Separate repeat native-reference drift from DT-versus-native error.

No sample is changed. Old/new measurements use identical frozen UID and token
identity but not identical execution shapes/lifecycle. Counts of sign changes
are descriptive, not a new numerical tolerance or proof of a kernel defect.
"""
import hashlib
import json
import math
from pathlib import Path

from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(REPO)).replace('\\', '/'),
                sha256=hashlib.sha256(raw).hexdigest())


def describe(points):
    return dict(points=len(points), trajectories=len({p['traj_uid'] for p in points}),
        states=len({p['initial_state_sha256'] for p in points}),
        absolute_d_drift=quantiles([abs(p['fresh_native_d']-p['old_native_d']) for p in points]),
        absolute_factual_score_drift=quantiles([abs(p['factual_score_drift']) for p in points]),
        absolute_deleted_score_drift=quantiles([abs(p['deleted_score_drift']) for p in points]),
        zero_sign_crossings=sum((p['fresh_native_d'] < 0) != (p['old_native_d'] < 0) for p in points),
        old_missed_negative_tail=sum(p['old_native_d'] < -math.log(2) and p['old_DT_d'] >= -math.log(2) for p in points),
        old_missed_tail_still_native_tail=sum(p['old_native_d'] < -math.log(2) and p['old_DT_d'] >= -math.log(2)
            and p['fresh_native_d'] < -math.log(2) for p in points),
        old_missed_tail_fresh_native_nonnegative=sum(p['old_native_d'] < -math.log(2) and p['old_DT_d'] >= -math.log(2)
            and p['fresh_native_d'] >= 0 for p in points))


def main():
    tasks = {}
    for task in ('textcraft', 'appworld'):
        paths = [HERE/f'layer-{task}-observations/rank{rank}.json' for rank in (0, 1)]
        cohorts = dict(uniform=[], predicted_tail_census=[])
        for path in paths:
            report = json.loads(path.read_bytes())
            assert report['phase'] == 'complete'
            for batch in report['batches']:
                for point in batch['points']:
                    for old in point['previous_comparisons']:
                        cohorts[old['cohort']].append(dict(
                            traj_uid=point['traj_uid'], packed_slot=point['packed_slot'],
                            token_id=point['token_id'], initial_state_sha256=point['initial_state_sha256'],
                            previously_examined=point['previously_examined'],
                            old_DT_d=old.get('saved_d', old.get('d')),
                            old_native_d=old['native_single_d'], fresh_native_d=point['native_single_d'],
                            old_native_stratum=stratum(old['native_single_d']),
                            fresh_native_stratum=stratum(point['native_single_d']),
                            factual_score_drift=point['factual_target_logp']-old['factual_target_logp'],
                            deleted_score_drift=point['deleted_target_logp']-old['deleted_target_logp']))
        data = {}
        for name, points in cohorts.items():
            cells = {}
            for p in points:
                cell = stratum(p['old_DT_d'])+':'+p['old_native_stratum']+':'+p['fresh_native_stratum']
                cells.setdefault(cell, []).append(p)
            data[name] = dict(summary=describe(points),
                old_DT_old_native_fresh_native_cells={k: describe(v) for k, v in sorted(cells.items())},
                by_state={k: describe([p for p in points if p['initial_state_sha256'] == k])
                    for k in sorted({p['initial_state_sha256'] for p in points})},
                points_with_identity=points)
        tasks[task] = dict(inputs=[ref(p) for p in paths], cohorts=data)
    result = dict(scope=__doc__, script=ref(Path(__file__)), tasks=tasks,
        model_calls=0, DT_calls=0, optimizer_steps=0, production_modified=False,
        conclusion='Native contrast values also drift; keep this uncertainty separate from finite DT error. No endpoint correction or tolerance change is justified by these counts.')
    output = REPO/'experiments/rl/results_native_reference_drift_20261008.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({t: {c: v['summary'] for c, v in s['cohorts'].items()} for t, s in tasks.items()}))


if __name__ == '__main__':
    main()
