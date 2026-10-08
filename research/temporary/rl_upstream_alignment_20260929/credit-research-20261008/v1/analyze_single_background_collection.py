"""Read-only paired error decomposition, separated by task/cohort/tail cells.

Joint-DT minus native single deletion equals: change of DT reference background,
single-DT propagation residual, and its native endpoint-score discrepancy. No
term is subtracted from training credit. No accuracy threshold is introduced.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from summarize_author_collection import stratum

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('textcraft','appworld'),required=True)
    args = parser.parse_args()
    stable_path = REPO/'experiments/rl/results_stable_negative_credit_20261008.json'
    stable = json.loads(stable_path.read_bytes())['tasks'][args.task]
    # Preserve the existing stabilized native precision/repetition intervals.
    baseline = {(p['traj_uid'],p['packed_slot']):p
                for cohort in stable['cohorts'].values() for p in cohort['points_with_identity']}
    folder = HERE/f'single-background-{args.task}-observations'
    reports = [json.loads((folder/f'rank{rank}.json').read_bytes()) for rank in (0,1)]
    points = []
    for report in reports:
        for batch in report['batches']:
            for p in batch['points']:
                base = baseline[p['traj_uid'],p['packed_slot']]
                joint, single, root = base['fresh_DT_d'],p['single_background_DT_d'],p['single_background_root']
                low, high = base['native_d_interval']
                native = statistics.mean(base['native_d_values'])
                components = dict(joint_minus_single_background=joint-single,
                    single_finite_minus_its_root=single-root,
                    single_root_minus_native_reference=root-native)
                point = dict(traj_uid=p['traj_uid'],packed_slot=p['packed_slot'],token_id=p['token_id'],
                    initial_state_sha256=p['initial_state_sha256'],previously_examined=p['previously_examined'],
                    cohorts=p['cohorts'],joint_d=joint,single_d=single,single_root=root,
                    native_interval=[low,high],native_mean=native,
                    joint_bin=stratum(joint),single_bin=stratum(single),native_bin=base['FP32_head_ratio_bin'],
                    original_spurious_tail=base['spurious_DT_tail_all_references'],
                    original_missed_tail=base['missed_native_tail_all_references'],
                    single_still_spurious_tail=(single < -math.log(2) and low >= 0),
                    single_opposite_sign=(single < 0 < low or single > 0 > high),
                    single_distance_to_reference_interval=max(low-single,single-high,0),
                    joint_distance_to_reference_interval=max(low-joint,joint-high,0),
                    components=components,
                    telescoping_roundoff=sum(components.values())-(joint-native),
                    single_factual_logp=p['single_background_factual_logp'],
                    single_deleted_logp=p['single_background_deleted_logp'],
                    artifact=p['artifact'])
                points.append(point)
    cells = {}
    for p in points:
        for cohort in p['cohorts']:
            key = '|'.join([cohort,'examined' if p['previously_examined'] else 'unexamined',p['joint_bin'],p['native_bin']])
            cells.setdefault(key,[]).append(p)
    summaries = {}
    for key,rows in cells.items():
        states = {}
        for p in rows:
            states.setdefault(p['initial_state_sha256'],{}).setdefault(p['traj_uid'],[]).append(p)
        def state_means(field):
            return [statistics.mean(statistics.mean(q['components'][field] for q in trajectory)
                                    for trajectory in trajectories.values()) for trajectories in states.values()]
        summaries[key] = dict(points=len(rows),trajectories=len({p['traj_uid'] for p in rows}),states=len(states),
            single_opposite_sign=sum(p['single_opposite_sign'] for p in rows),
            original_spurious_tail=sum(p['original_spurious_tail'] for p in rows),
            single_still_spurious_tail=sum(p['single_still_spurious_tail'] for p in rows),
            components={field:dict(equal_state_mean=statistics.mean(state_means(field)),
                point_median=statistics.median(p['components'][field] for p in rows),
                point_min=min(p['components'][field] for p in rows),point_max=max(p['components'][field] for p in rows))
                for field in points[0]['components']},
            joint_distance_median=statistics.median(p['joint_distance_to_reference_interval'] for p in rows),
            single_distance_median=statistics.median(p['single_distance_to_reference_interval'] for p in rows))
    result = dict(scope=__doc__,task=args.task,complete=all(r['phase']=='complete' for r in reports),
        inputs=[ref(stable_path)]+[ref(folder/f'rank{rank}.json') for rank in (0,1)],points=points,cells=summaries,
        point_count=len(points),original_spurious_tail=sum(p['original_spurious_tail'] for p in points),
        original_spurious_tail_still_spurious=sum(p['original_spurious_tail'] and p['single_still_spurious_tail'] for p in points),
        max_telescoping_roundoff=max(abs(p['telescoping_roundoff']) for p in points),
        operations_per_rank=[r['operations'] for r in reports],seconds_per_rank=[r['elapsed_seconds'] for r in reports],
        production_modified=False,credit_repaired=False,official_tolerance_claim=False,
        weighting='Within each task/cohort/exposure/prediction-reference cross-cell only: source mean per trajectory, trajectory mean per initial state, equal state mean. Tail and body raw moments never pooled.',
        interpretation='Diagnostic reference changes do not provide a cost-compliant training repair, new estimator, whole-method faithfulness claim, or evidence of training degradation.')
    path = HERE/f'single-background-{args.task}-analysis.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['points','cells','inputs']},ensure_ascii=False))


if __name__ == '__main__':
    main()
