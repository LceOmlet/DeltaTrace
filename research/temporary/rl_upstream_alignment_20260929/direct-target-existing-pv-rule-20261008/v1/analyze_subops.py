"""Bind passive decoder branch measurements to the completed full diagnosis.

No attribution, model operation, repair, or acceptance tolerance is defined here.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def artifact(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


launch = json.loads((HERE / 'subops-results/launch.json').read_bytes())
assert launch['partial_DT_only'] and not launch['full_signed_vector_produced']
records = []
for rank in (0, 1):
    path = HERE / 'subops-results/results' / f'rank{rank}.json'
    current = json.loads(path.read_bytes())
    previous = json.loads((HERE / 'layer-results/results' / f'rank{rank}.json').read_bytes())
    assert current['phase'] == 'complete'
    assert current['geometry'] == previous['geometry']
    assert current['owners'] == previous['owners']
    points = []
    checks = []
    for layer in (31, 30):
        branch = current['decoder_subops'][str(layer)]
        values = branch['points']
        for index, value in enumerate(values):
            before = (values[index - 1]['coefficient_times_single_delta']
                      if index else value['coefficient_times_single_delta'])
            points.append(dict(decoder=layer, **value,
                change_from_preceding_point=value['coefficient_times_single_delta'] - before))
        checks.append(dict(decoder=layer, values=branch['factual_endpoint_checks']))
    boundary_comparison = []
    for key, value in current['cross_boundary_contractions'].items():
        old = previous['cross_boundary_contractions'][key]
        fields = ('joint_coefficient_times_single_deletion_delta',
                  'joint_coefficient_times_joint_delta')
        boundary_comparison.append(dict(boundary=int(key),
            equal_to_previous={name: value[name] == old[name] for name in fields},
            differences={name: value[name] - old[name] for name in fields},
            current=value))
    phases = [json.loads(line) for line in path.with_name(f'rank{rank}-phases.jsonl').read_text().splitlines()]
    records.append(dict(rank=rank, worker_pid=current['pid'], worker_birth=current['birth'],
        points=points, factual_endpoint_checks=checks, boundary_comparison=boundary_comparison,
        phases=current['phases'], max_recorded_pss_bytes=max(v['pss_bytes'] for v in phases),
        completed_unix=current['unix'], source=artifact(path), finite_owner=current['finite_owner']))

out = dict(status='Completed passive decoder31/30 branch diagnosis; no credit repair deployed',
    diagnostic_code_commit=launch['code_commit'], source_sha256=launch['source_sha256'],
    native_sha256=launch['native_sha256'], case_binding=launch['case_binding'],
    ranks=records, points_equal_across_ranks=records[0]['points'] == records[1]['points'],
    elapsed_launch_to_last_worker_complete_seconds=max(v['completed_unix'] for v in records) - launch['launched_unix'],
    sources=[artifact(HERE / 'subops-results/transport.json'),
             artifact(HERE / 'subops-results/results/effective-config.yaml'),
             artifact(HERE / 'subops-producer-owner.json')],
    scope='Two partial original producer DT calls per rank, stopped after decoder30. Original native endpoint hooks and original token-effect contractions only. No full signed vector or advantages produced.',
    operations=dict(partial_DT_calls_per_rank=2, backward=0, optimizer=0, rollout=0, checkpoint_restore=0),
    limitations=[
        'A change in a joint coefficient contracted with a single-deletion state difference identifies a propagation boundary. It does not alone prove a faulty numerical kernel.',
        'Exact comparisons to the earlier same-B4 boundary results check that passive instrumentation did not alter those results. They are not FA/FLA precision acceptance tests.',
        'Physical VRAM observations are separately saved polls; Torch allocator and recorded PSS values do not establish a continuous physical peak.',
        'This partial run produces neither full token attribution nor PPO advantages and cannot establish repaired training quality.',
    ],
    credit_repaired=False, production_profile_changed=False, official_tolerance_claim=False,
    formal_restart=False, text_update_released=False)
target = REPO / 'experiments/rl/results_current_extreme_subops_20261008.json'
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(dict(receipt=artifact(target), ranks=[dict(rank=v['rank'], points=v['points'],
    boundary_comparison=[dict(boundary=b['boundary'], equal=b['equal_to_previous'], differences=b['differences'])
                         for b in v['boundary_comparison']]) for v in records]), indent=2))
