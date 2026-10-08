"""Compare existing native runs with the same frozen query identity and layout.

The layer observer run included a preceding DT and passive decoder hooks;
the head run did not. A difference is observed, not attributed to either
factor until the bounded lifecycle/observer isolation has measured them.
"""
import hashlib
import json
from pathlib import Path

from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    return dict(path=str(path.relative_to(REPO)).replace('\\', '/'),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def describe(points):
    return dict(points=len(points), trajectories=len({p['traj_uid'] for p in points}),
        states=len({p['initial_state_sha256'] for p in points}),
        abs_layer_native_to_head_native_d=quantiles([abs(p['layer_native_d']-p['head_native_d']) for p in points]),
        head_native_exactly_matches_first_native_d=sum(p['head_native_d'] == p['first_native_d'] for p in points),
        head_native_exactly_matches_layer_native_d=sum(p['head_native_d'] == p['layer_native_d'] for p in points),
        DT_source_contraction_exactly_matches_selected_embedding=sum(p['DT_d'] == p['embedding_contraction'] for p in points),
        first_factual_embedding_identical=sum(p['factual_embedding_identical'] for p in points))


def main():
    tasks = {}
    for task in ('textcraft', 'appworld'):
        layer_paths = [HERE/f'layer-{task}-observations/rank{rank}.json' for rank in (0, 1)]
        head_paths = [HERE/f'native-head-{task}-final/rank{rank}.json' for rank in (0, 1)]
        layer, head = {}, {}
        for paths, target in ((layer_paths, layer), (head_paths, head)):
            for path in paths:
                value = json.loads(path.read_bytes())
                assert value['phase'] == 'complete'
                for batch in value['batches']:
                    for point in batch['points']:
                        key = point['traj_uid'], point['packed_slot']
                        assert key not in target
                        target[key] = point
        assert layer.keys() == head.keys() and len(head) == 165
        cohorts = dict(uniform=[], predicted_tail_census=[])
        for key, new in head.items():
            old = layer[key]
            assert old['token_id'] == new['token_id']
            for first in new['previous_comparisons']:
                cohorts[first['cohort']].append(dict(traj_uid=key[0], packed_slot=key[1],
                    initial_state_sha256=new['initial_state_sha256'],
                    first_native_d=first['native_single_d'], layer_native_d=old['native_single_d'],
                    head_native_d=new['scores']['native']['d'],
                    DT_d=old['fresh_DT_d'], embedding_contraction=old['boundaries']['0']['value'],
                    factual_embedding_identical=old['boundaries']['0']['factual_endpoints_equal'],
                    old_DT_d=first.get('saved_d', first.get('d'))))
        data = {}
        for cohort, points in cohorts.items():
            grouped = {}
            for point in points:
                cell = stratum(point['old_DT_d'])+':'+stratum(point['first_native_d'])
                grouped.setdefault(cell, []).append(point)
            data[cohort] = dict(summary=describe(points),
                original_DT_native_cells={k: describe(v) for k, v in sorted(grouped.items())},
                points_with_identity=points)
        tasks[task] = dict(inputs=[ref(p) for p in layer_paths+head_paths], cohorts=data)
    result = dict(scope=__doc__, script=ref(Path(__file__)), tasks=tasks, model_calls=0,
        DT_calls=0, optimizer_steps=0, production_modified=False,
        interpretation=[
            'Identical source embedding contraction excludes a source-position mismatch at this observed boundary, not every downstream actor interface.',
            'Previous attribution of all native score drift to batch shape was unproven. Preceding DT, observer hooks and precision/runtime state require controlled separation.',
            'No residual is subtracted, no credit is replaced, and no acceptance tolerance is invented.'
        ])
    output = REPO/'experiments/rl/results_native_reference_repeat_20261008.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({t: {c: v['summary'] for c, v in s['cohorts'].items()} for t, s in tasks.items()}))


if __name__ == '__main__':
    main()
