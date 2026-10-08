"""Summarize a bounded observation of the unchanged VERL actor, without a pass claim."""
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(REPO)).replace('\\', '/'),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    launch = json.loads((HERE/'native-nonfinite-launch.json').read_bytes())
    observations = [json.loads(p.read_bytes()) for p in
                    sorted((HERE/'native-nonfinite-observations').glob('*.json'),
                           key=lambda p: int(p.stem))]
    final = observations[-1]
    assert not final['processes'], 'This summary requires confirmed terminal processes'
    assert final['launch']['pid'] == launch['pid']
    assert final['launch']['birth'] == launch['birth']
    ranks = []
    for rank in (0, 1):
        path = HERE/'native-nonfinite-final'/f'rank{rank}.json'
        report = json.loads(path.read_bytes())
        steps = report['optimizer_steps']
        ranks.append(dict(rank=rank, report=ref(path), actor=report['actor'],
            policy=report['policy'], microbatches_started=len(report['microbatches']),
            planned_microbatches=32,
            output_gradients_observed=sum('log_prob_gradient' in m for m in report['microbatches']),
            completed_optimizer_steps=sum('completed_unix' in s for s in steps),
            planned_optimizer_steps=4,
            native_grad_norms=[s.get('native_grad_norm') for s in steps],
            pre_native_step_nonfinite_gradient_elements=[
                sum(v['nonfinite'] for v in s['gradients'].values()) for s in steps],
            last_microbatch_index=report['microbatches'][-1]['index'],
            last_recorded_phase=report['microbatches'][-1]['phase'],
            exception=report.get('error')))
        assert all(math.isfinite(s['native_grad_norm']) for s in steps if 'native_grad_norm' in s)
    log = (HERE/'native-nonfinite-final/driver.log').read_text(errors='replace')
    log_copy = HERE/'native-nonfinite-final/driver-log.txt'
    log_copy.write_bytes((HERE/'native-nonfinite-final/driver.log').read_bytes())
    assert 'SIGTERM received' in log
    phase_times = {}
    for line in log.splitlines():
        if line.startswith('{"phase":'):
            row = json.loads(line)
            phase_times[row['phase']] = row['unix']
    allocated = []
    for obs in observations:
        for match in re.finditer(r'\|\s*[45]\s+MetaX C550.*?\|\s*[^\n]*?\|\s*(\d+)/65536 MiB',
                                 obs['physical'], re.S):
            allocated.append(int(match.group(1)))
    result = dict(
        status='partial_terminated_by_900_second_diagnostic_wall_budget',
        fixed=False, full_update_passed=False, observed_unix=final['observed_unix'],
        scope='Separate native actor NaN investigation; not extreme-attribution causality or training-quality assessment.',
        launch=ref(HERE/'native-nonfinite-launch.json'),
        script=ref(HERE/'inspect_native_actor_nonfinite.py'),
        driver_log=ref(log_copy),
        launch_pid=launch['pid'], launch_birth=launch['birth'],
        actual_source_sha256=launch['source_sha256'],
        code_base_commit=launch['diagnostic_code_commit'], devices=launch['devices'],
        preserved_configuration=dict(rows=256, global_optimizer_minibatch=64,
            per_card_microbatch=4, lora_rank=8, lora_alpha=16,
            loss='Original VERL policy, entropy, KL, clipping, optimizer and scheduler'),
        official_diagnostic='torch.autograd.detect_anomaly(check_nan=True); unchanged returned gradients',
        ranks=ranks,
        old_log_prob_seconds=phase_times['native_ref_log_prob_begin']-phase_times['native_old_log_prob_begin'],
        reference_log_prob_seconds=phase_times['native_update_begin']-phase_times['native_ref_log_prob_begin'],
        wall_budget_seconds=900,
        terminal_observation=ref(HERE/'native-nonfinite-observations'/f"{int(final['observed_unix'])}.json"),
        remaining_owned_processes=0,
        maximum_observed_process_tree_pss_bytes=max(sum(p['pss_bytes'] for p in o['processes']) for o in observations),
        maximum_observed_physical_gpu_MiB=max(allocated) if allocated else None,
        physical_gpu_limit_MiB=65536,
        DT_calls=0, rollouts=0, checkpoint_restore=0, formal_training_restarted=False,
        evidence_limit=[
            'First three sequential native optimizer minibatches were finite; fourth did not complete.',
            'Original LoRA A initialization and old/ref outputs were not saved; fresh reproduction is not bitwise historical replay.',
            'This fresh actor did not execute the preceding DT lifecycle of the failed formal process.',
            'Anomaly detection and synchronous per-parameter observations add overhead; timing is not formal training throughput.',
            'No cause located and no repair or new tolerance introduced. Do not repeat the same full replay merely to expand the count.'
        ],
        interpretation='Saved DT coefficients alone did not reproduce the original NaN in the observed three native optimizer minibatches. This does not establish that the full original update or DT-to-actor lifecycle is correct.')
    output = REPO/'experiments/rl/results_native_actor_nonfinite_20261008.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], ranks=[
        dict(rank=r['rank'], completed_steps=r['completed_optimizer_steps'],
             started_microbatches=r['microbatches_started']) for r in ranks],
        remaining_owned_processes=0, output=str(output))))


if __name__ == '__main__':
    main()
