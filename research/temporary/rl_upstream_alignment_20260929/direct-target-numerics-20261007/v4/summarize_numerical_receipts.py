"""Summarize exact completed native receipts; no tensor/model operations."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'


def read(path):
    return json.loads(path.read_bytes())


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


inspection = read(RAW / 'native-owner-inspection.json')
job = read(RAW / 'job.json')
completed = read(RAW / 'official-checks-completed.json')
release = read(RAW / 'final-release.json')
result = dict(
    scope='One original observed B4 per rank, eight saved original current-response inputs; '
          'the pack has two trajectory UIDs across different turns, not eight complete joint trajectories',
    candidate=inspection['candidate'], owners=inspection['sources'],
    native_job=job, original_launcher_runtime=inspection['launcher_runtime'],
    source_inputs=inspection['source'], input_identity=inspection['input'],
    ranks={}, checks=dict(FA=[], FLA=[]),
    operations=dict(original_B4_attribution_calls_per_rank=1, real_input_rows_global=8,
                    paired_endpoint_rows_per_rank=8, model_initializations_per_rank=1,
                    rollout=0, checkpoint=0, optimizer=0, training_backward=0),
    limitations=[
        'The original single-response executed-payload pack is a multi-token numerical input; '
        'it does not establish full multi-round joint-trajectory or 32768-token head capacity.',
        'FA native output/gradient and coincident finite suffix checks use the original seven '
        'assertions; they do not bound noncoincident full-network finite attribution error.',
        'FLA checks cover original forward o and final state ht with actual cached initial state, '
        'not FLA input adjoints or whole DT tolerances.',
        'The actual full-vocabulary text seed returned through its unchanged owner guard; '
        'no independent actual full-vocabulary multi-target head reference assertion is claimed.',
        'Observed elapsed time includes cold initialization/compilation, passive hooks and CPU '
        'operand exports; it is not a warm throughput comparison.',
        'Raw self/prior advantages are original report descriptions; no whitening, actor PPO '
        'update, task-quality recovery or gradient-effect claim is made.',
    ],
)
for rank in range(2):
    path = RAW / f'rank{rank}.json'
    original = read(path)
    observation = original['reports'][0]
    report = observation['report']
    geometry = inspection['ranks'][rank]
    worker_path = RAW / f'actual-worker-rank{rank}.stdout.txt'
    phases = [json.loads(line.partition('[direct-target phase] ')[2])
        for line in worker_path.read_text(errors='replace').splitlines()
        if line.startswith('[direct-target phase] ')]
    result['ranks'][str(rank)] = dict(
        phase=original['phase'], geometry=geometry, report_identity=identity(path),
        report_counts={key: report[key] for key in ('finite_trace_calls', 'joint_target_requests',
            'policy_tokens', 'target_self_tokens', 'prior_source_tokens', 'query_tokens',
            'synthetic_labels', 'prefix_lease_used')},
        raw_advantage_groups=report['raw_advantage_groups'],
        original_endpoint_traces=report['traces'],
        observed_seconds=observation['seconds'],
        torch_peak_allocated_bytes=observation['torch_peak_allocated_bytes'],
        torch_peak_reserved_bytes=observation['torch_peak_reserved_bytes'],
        actual_head_and_seed_observations=observation['head_and_seed_observations'],
        original_fulltext_seed_returned=any(item['phase'] == 'finite_text_seed_end' for item in phases),
        native_phase_count=len(phases), phase_log=identity(worker_path),
        saved_vectors=observation['vectors'], actual_saved_operands=observation['actual_operands'],
    )
    for sample in range(4):
        path = RAW / f'official-checks-rank{rank}' / f'fa3-sample{sample}.json'
        check = read(path)
        result['checks']['FA'].append(dict(rank=rank, sample=sample, source=identity(path),
            status=check['status'], query_shape=check['query_shape'], key_shape=check['key_shape'],
            operand_dtypes=check['operand_dtypes'], assertions=check['assertions'], checks=check['checks'],
            query_start=check['query_start'], coefficient_starts=check['coefficient_starts']))
    path = RAW / f'official-checks-rank{rank}' / 'gdn0.json'
    check = read(path)
    result['checks']['FLA'].append(dict(rank=rank, source=identity(path),
        status=check['status'], original_assertions=check['original_assertions'],
        actual_calls=check['actual_calls'], actual_tensor_metadata=check['actual_tensor_metadata'],
        normalization_applied_once=check['normalization_applied_once'],
        initial_state_was_none=check['initial_state_was_none'],
        official_source=check['official_source']))
fa = [item for record in result['checks']['FA'] for item in record['checks']]
result['assertion_counts'] = dict(FA_total=len(fa), FA_passed=sum(item['status'] == 'passed' for item in fa),
    FLA_total=sum(len(item['original_assertions']) for item in result['checks']['FLA']),
    FLA_passed=sum(len(item['original_assertions']) for item in result['checks']['FLA'] if item['status'] == 'passed'))
result['offline_completed'] = completed
result['release_after_all_checks'] = release
result['sources'] = {str(path.relative_to(HERE)): identity(path)
    for path in sorted(RAW.rglob('*.json'))}
result['summary_source'] = identity(Path(__file__))
path = HERE / 'direct-target-numerical-summary.json'
path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
compact = dict(detail_summary=identity(path), summary_source=result['summary_source'],
    scope=result['scope'], candidate=result['candidate'], owners=result['owners'],
    input_identity=result['input_identity'], operations=result['operations'],
    assertion_counts=result['assertion_counts'],
    original_FA_assertions=result['checks']['FA'][0]['assertions'],
    original_FLA_assertions=result['checks']['FLA'][0]['original_assertions'],
    ranks={}, limitations=result['limitations'],
    completed_source=identity(RAW / 'official-checks-completed.json'),
    released_source=identity(RAW / 'final-release.json'),
    release_observed_unix=release['observed_unix'], recorded_pids_present=release['recorded_pids_present'])
for rank, record in result['ranks'].items():
    compact['ranks'][rank] = {key: record[key] for key in ('phase', 'geometry', 'report_identity',
        'report_counts', 'raw_advantage_groups', 'observed_seconds', 'torch_peak_allocated_bytes',
        'torch_peak_reserved_bytes', 'original_fulltext_seed_returned', 'native_phase_count', 'phase_log')}
    compact['ranks'][rank]['FA_rows'] = [dict(sample=item['sample'], source=item['source'],
        status=item['status'], checks=len(item['checks']), query_shape=item['query_shape'],
        key_shape=item['key_shape'], operand_dtypes=item['operand_dtypes'])
        for item in result['checks']['FA'] if str(item['rank']) == rank]
    compact['ranks'][rank]['FLA_call'] = {key: result['checks']['FLA'][int(rank)][key]
        for key in ('source', 'status', 'actual_calls', 'normalization_applied_once',
                    'initial_state_was_none', 'official_source')}
compact_path = HERE / 'direct-target-numerical-receipt.json'
compact_path.write_text(json.dumps(compact, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    compact_path=str(compact_path), compact_sha256=hashlib.sha256(compact_path.read_bytes()).hexdigest(),
    assertion_counts=result['assertion_counts'], release_unix=release['observed_unix'])))
