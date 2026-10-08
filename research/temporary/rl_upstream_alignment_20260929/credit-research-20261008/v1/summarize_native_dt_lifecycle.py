"""Report original native-forward preservation, not training or attribution quality."""
import hashlib
import json
from pathlib import Path

from summarize_author_collection import quantiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    return dict(path=str(path.relative_to(REPO)).replace('\\', '/'),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    folder = HERE/'native-dt-lifecycle-textcraft-final'
    paths = [folder/f'rank{rank}.json' for rank in (0, 1)]
    ranks = [json.loads(p.read_bytes()) for p in paths]
    assert all(r['phase'] == 'complete' for r in ranks)
    points = []
    for rank in ranks:
        for batch in rank['batches']:
            probes = {p['name']: p for p in batch['lifecycle_probes']}
            before = probes['before_DT_repeat_1']
            assert len(probes) == 4
            assert all(p['input_sha256'] == before['input_sha256'] for p in probes.values())
            by_uid = {p['traj_uid']: p for p in batch['points']}
            for row, uid in enumerate(batch['uids']):
                base_fact = before['factual_logp'][row]
                base_d = base_fact-before['deleted_logp'][row]
                comparisons = {}
                for name, probe in probes.items():
                    comparisons[name] = dict(factual_delta=probe['factual_logp'][row]-base_fact,
                        d_delta=probe['factual_logp'][row]-probe['deleted_logp'][row]-base_d,
                        flags_unchanged=probe['before_flags'] == before['before_flags']
                            and probe['after_flags'] == before['after_flags'])
                observed = by_uid[uid]
                comparisons['with_original_layer_observer_hooks'] = dict(
                    factual_delta=observed['factual_target_logp']-base_fact,
                    d_delta=observed['native_single_d']-base_d)
                points.append(dict(traj_uid=uid, initial_state_sha256=observed['initial_state_sha256'],
                    packed_slot=observed['packed_slot'], batch=batch['index'], rank=rank['rank'],
                    comparisons=comparisons))
    assert len(points) == 45 and len({p['traj_uid'] for p in points}) == 45
    summary = {}
    for phase in points[0]['comparisons']:
        items = [p['comparisons'][phase] for p in points]
        summary[phase] = dict(trajectories=len(items),
            absolute_factual_delta=quantiles([abs(v['factual_delta']) for v in items]),
            absolute_d_delta=quantiles([abs(v['d_delta']) for v in items]),
            unchanged_flag_records=sum(v.get('flags_unchanged', False) for v in items)
                if all('flags_unchanged' in v for v in items) else None)
    result = dict(scope=__doc__, script=ref(Path(__file__)), inputs=[ref(p) for p in paths],
        source_sha256=ranks[0]['source_sha256'], frozen_plan_sha256=ranks[0]['input_plan_sha256'],
        trajectories=len(points), states=len({p['initial_state_sha256'] for p in points}),
        frozen_B4_groups=sum(len(r['batches']) for r in ranks),
        query_rule='The already fixed first query of every frozen B4 row; preservation diagnostic, not a replacement quality sample.',
        phases=summary, points_with_identity=points,
        operations=[r['operations'] for r in ranks], elapsed_seconds=[r['elapsed_seconds'] for r in ranks],
        production_modified=False,
        limitations=[
            'Observed only the recorded native-forward boundary; this does not prove DT-to-PPO gradient lifecycle correctness.',
            'This uses the current diagnostic import path. Older-run path differences require a separate source comparison.',
            'No official numerical tolerance or attribution-quality pass is inferred from forward preservation.'
        ])
    output = REPO/'experiments/rl/results_native_dt_lifecycle_20261008.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), trajectories=result['trajectories'], states=result['states'],
        phases=summary, elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
