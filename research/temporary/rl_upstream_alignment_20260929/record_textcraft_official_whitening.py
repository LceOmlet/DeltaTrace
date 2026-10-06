"""Bind completed official-whitening observations without changing old receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from prepare_textcraft_official_whitening import AUDIT, LOCAL, REPO


def source(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


if __name__ == '__main__':
    analysis = json.loads((LOCAL / 'official-whitening-analysis.json').read_bytes())
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    assert completed['optimizer_steps'] == completed['scheduler_steps'] == 0
    assert completed['DT_calls'] == completed['sampling_calls'] == 0
    assert analysis['normalization']['official_helper_exact_equal']
    assert all(r['group_config_exact'] and r['input_shapes_and_dtypes_exact'] for r in analysis['per_rank'])
    for r in analysis['per_rank']:
        assert r['operations']['optimizer_step_executed'] == [False, False]
        assert r['operations']['scheduler_step_executed'] == [False, False]
    job = json.loads((LOCAL / 'job.json').read_bytes())
    candidate = json.loads((LOCAL / 'candidate-source.json').read_bytes())
    runtime_path = LOCAL / 'current_runtime.json'
    runtime = json.loads(runtime_path.read_bytes())
    revision = runtime['local_patch_commit']
    # Bind the staged patch bytes to their immutable committed owner source.
    patch = REPO / 'experiments/rl/patch_verl_agent2.py'
    committed = subprocess.run(['git', 'show', revision + ':experiments/rl/patch_verl_agent2.py'],
                               capture_output=True, check=True, cwd=REPO).stdout
    assert hashlib.sha256(committed).hexdigest() == source(patch)['sha256']
    runtime.update(observed_unix=time.time(), status='cpu_and_native_gradient_verified_not_formally_deployed',
        job=dict(pid=job['pid'], pid_birth=job['pid_birth'], started_unix=job['started_unix'],
                 completed_unix=completed['completed_unix'], devices=job['devices']),
        native_results_pending=False, native_result=source(LOCAL / 'official-whitening-analysis.json'))
    runtime['verification_receipts']['independent_owner_interface_review'] = source(
        LOCAL / 'independent-owner-interface-review.json')
    runtime['verification_receipts']['independent_native_gradient_review'] = source(
        LOCAL / 'independent-native-gradient-review.json')
    runtime_path.write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
    record = dict(
        scope='Approved actor-advantage preprocessing only; raw EOS DT Q/V/A and upstream PPO retained.',
        implementation_patch_commit=revision,
        upstream_commit=candidate['upstream_commit'],
        status='implemented_and_CPU_native_gradient_tested_not_formal_training_recovery',
        sources={name: source(path) for name, path in {
            'method': REPO / 'experiments/rl/PLAN.md',
            'source_patch': patch,
            'focused_owner_tests': REPO / 'tests/test_dt_official_whitening.py',
            'collector_tests': REPO / 'experiments/rl/test_rollout_credit.py',
            'candidate_preparer': AUDIT / 'prepare_textcraft_official_whitening.py',
            'diagnostic': AUDIT / 'verify_textcraft_official_whitening.py',
            'stager': AUDIT / 'stage_textcraft_official_whitening.py',
            'analyzer': AUDIT / 'analyze_textcraft_official_whitening.py',
            'recorder': Path(__file__),
            'candidate_source': LOCAL / 'candidate-source.json',
            'focused_test_output': LOCAL / 'focused-owner-tests.stdout.txt',
            'collector_test_output': LOCAL / 'collector-owner-tests.stdout.txt',
            'collector_test_receipt': LOCAL / 'collector-owner-test-receipt.json',
            'collector_junit': LOCAL / 'collector-owner-tests.xml',
            'runtime': runtime_path,
            'independent_CPU_review': LOCAL / 'independent-owner-interface-review.json',
            'independent_native_gradient_review': LOCAL / 'independent-native-gradient-review.json',
            'unchanged_owner_effective_config': LOCAL / 'base-owner-effective-config.yaml',
            'analysis': LOCAL / 'official-whitening-analysis.json',
        }.items()},
        candidate_source=candidate,
        actual_native_analysis=analysis,
        completed_job=dict(pid=job['pid'], pid_birth=job['pid_birth'], started_unix=job['started_unix'],
            completed_unix=completed['completed_unix'], wall_seconds=completed['completed_unix']-job['started_unix'],
            devices=job['devices']),
        resource_observations=[source(p) for p in sorted(LOCAL.glob('observation-*.json'))],
        official_implementation='Pinned VERL masked_whiten default shift_mean=True, called once on full collected action mask before native split; output remasked. No copied normalizer or new tolerance.',
        unchanged='Raw Q/V/A, observed returns, old/ref logprob, action/observation masks, actor/core/worker bytes, configuration, LoRA8/16 and per-card B4.',
        limits='This measures task-gradient scale and paired direction; no real optimizer update, success-rate recovery or DT attribution-quality claim.',
        formal_training='TextCraft remains stopped; no formal task restarted or deployment altered.')
    target = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    result = json.loads(target.read_bytes())
    # The old PLAN bytes remain in Git, not in another competing method file.
    old_commit = '9a9b53240152a12bd4127ebc2afd0090e87bf65d'
    old = subprocess.run(['git', 'show', old_commit + ':experiments/rl/PLAN.md'],
                         capture_output=True, check=True, cwd=REPO).stdout
    old_sha = hashlib.sha256(old).hexdigest()

    def preserve_historical_plan(value):
        if isinstance(value, dict):
            if value.get('path', '').replace('\\', '/').endswith('experiments/rl/PLAN.md') and value.get('sha256') == old_sha:
                value.pop('path')
                value.update(source_kind='Historical Git blob; not the current accepted method',
                    git_commit=old_commit, repository_path='experiments/rl/PLAN.md')
            for child in value.values():
                preserve_historical_plan(child)
        elif isinstance(value, list):
            for child in value:
                preserve_historical_plan(child)

    preserve_historical_plan(result)
    result['official_advantage_whitening'] = record
    result['status'] = 'weak_task_signal_official_whitening_native_gradient_test_completed_learning_recovery_not_deployed'
    result['updated_unix'] = time.time()
    result['production_status']['method'] = 'Raw EOS DT Q/V/A preserved; user-approved official actor whitening implemented/tested in isolated candidate, not formally deployed'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(result=source(target), runtime=source(runtime_path), patch_commit=revision,
                         job=record['completed_job']), ensure_ascii=False))
