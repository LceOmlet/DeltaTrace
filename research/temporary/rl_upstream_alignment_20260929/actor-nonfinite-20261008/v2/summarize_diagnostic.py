"""Summarize saved actual diagnostic observations without running a model."""
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def receipt(path):
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    observations = [json.loads(p.read_bytes()) for p in sorted((HERE/'observations').glob('[0-9]*.json'))]
    final = observations[-1]
    files = final['files']
    phase = files['phase.json']
    assert phase['phase'] in ('complete', 'failed'), 'Do not mark an active or timed-out run complete'
    ranks = [files[f'rank{i}.json'] for i in (0, 1)]
    sources = json.loads((HERE/'lifecycle-sources.json').read_bytes())
    for source in sources:
        if source.get('launch_expected_sha256'):
            assert source['sha256'] == source['launch_expected_sha256']
    identity = json.loads((HERE/'original-PPO-function-identity.json').read_bytes())
    assert all(x['function_AST_identical'] for x in identity['functions'])
    summaries = []
    for rank in ranks:
        completed = [s for s in rank['optimizer_steps'] if s.get('completed_unix')]
        micro = rank['microbatches']
        nonfinite = []
        for row in micro:
            for key, value in row.items():
                if isinstance(value, dict) and value.get('nonfinite', 0):
                    nonfinite.append(dict(microbatch=row['index'], field=key, value=value))
                if key == 'policy_outputs':
                    nonfinite.extend(dict(microbatch=row['index'], field=key, value=x)
                                     for x in value if x['nonfinite'])
        summaries.append(dict(rank=rank['rank'], phase=rank['phase'],
            microbatches_started=len(micro), planned_microbatches=32,
            completed_optimizer_steps=len(completed), planned_optimizer_steps=4,
            native_grad_norms=[s['native_grad_norm'] for s in completed],
            pre_step_nonfinite_gradients=[s['nonfinite_gradients'] for s in rank['optimizer_steps']],
            observed_gradient_tensors_per_step=[s['observed_gradient_tensors'] for s in completed],
            nonfinite_observed_outputs=nonfinite, exception=rank.get('error'),
            traceback=rank.get('traceback'),
            report=receipt(HERE/'observations'/f"rank{rank['rank']}.json")))
    full_update = (phase['phase'] == 'complete' and all(
        r['phase'] == 'complete' and r['completed_optimizer_steps'] == 4 and
        r['microbatches_started'] == 32 and not r['exception'] and
        all(math.isfinite(n) for n in r['native_grad_norms']) and
        not r['nonfinite_observed_outputs'] and not any(r['pre_step_nonfinite_gradients'])
        for r in summaries))
    changes = []
    for i in (0, 1):
        a, b = (files[f'rank{i}-{stage}.json'] for stage in ('before_DT', 'after_DT'))
        differences = {k:[a[k], b[k]] for k in a.keys() & b.keys()
                       if a[k] != b[k] and k not in ('phase', 'unix', 'allocated', 'reserved', 'pss_bytes')}
        changes.append(dict(rank=i, recorded_state_changes=differences))
    stage_times = {}
    for observation in observations:
        stage = observation['files'].get('phase.json', {})
        if stage.get('phase'):
            stage_times.setdefault(stage['phase'], stage['unix'])
    pairs = [('native_old_log_prob_begin', 'native_ref_log_prob_begin'),
             ('native_ref_log_prob_begin', 'original_DT_begin'),
             ('original_DT_begin', 'native_update_begin'), ('native_update_begin', 'complete')]
    times = {a:stage_times[b]-stage_times[a] for a,b in pairs if a in stage_times and b in stage_times}
    memory = [int(x) for o in observations for x in re.findall(r'(\d+)/65536 MiB',o['physical'])]
    latest = HERE/'observations'/f"{int(final['observed_unix'])}.json"
    result = dict(
        status='fresh_DT_then_full_native_update_finite' if full_update else 'native_diagnostic_failed',
        root_cause_located=False, fixed=False, observed_unix=final['observed_unix'],
        scope='TextCraft PPO nonfinite gradients only; separate from extreme attribution and historical entropy degradation.',
        original_incident='Both original ranks reported one nonfinite norm; native VERL skipped the affected step. Original minibatch index is not known.',
        preceding_diagnostic='900-second diagnostic timed out during its fourth update, not a reproduced fourth-update NaN.',
        launch=final['launch'], upstream_commit='20bd331',
        diagnostic_worker=receipt(HERE/'diagnose_dt_to_actor.py'),
        original_function_identity=receipt(HERE/'original-PPO-function-identity.json'),
        deployed_lifecycle_sources=receipt(HERE/'lifecycle-sources.json'),
        original_input_audit=receipt(HERE/'saved-input-microbatch-audit.json'),
        terminal_observation=receipt(latest),
        preserved_configuration=dict(actor_rows=256, DT_rows=176,
            global_optimizer_minibatch=64, per_card_microbatch=4, lora_rank=8, lora_alpha=16,
            loss='Original VERL policy, entropy, KL, clipping, optimizer and scheduler',
            advantages='Exact saved original whole-batch whitened actor values; recomputed DT result not substituted'),
        diagnostics='Original PyTorch detect_anomaly(check_nan=True) plus passive returned-value/gradient observations; no new tolerance.',
        full_fresh_update_finite=full_update, ranks=summaries,
        DT_output=files.get('DT-output-observation.json'), DT_lifecycle_states=changes,
        retained_initial_LoRA=[files[f'rank{i}-initial.json']['initial_tensor_capture'] for i in (0,1)],
        retained_old_ref_input=files.get('actor-input-with-native-old-ref.json'),
        phase_seconds=times, maximum_observed_tree_pss_bytes=max(sum(p['pss_bytes'] for p in o['processes']) for o in observations),
        maximum_observed_physical_GPU_MiB=max(memory) if memory else None,
        remaining_owned_non_zombie_processes=[p['pid'] for p in final['processes'] if p['status']!='zombie'],
        formal_training_restarted=False, checkpoint_restore=False, deployed_repair=False,
        evidence_limits=[
            'Fresh actor reuses the saved original inputs, not lost historical initial LoRA A or old/ref log-probs.',
            'DT lifecycle is included, but the original preceding vLLM rollout lifecycle is not replayed.',
            'Diagnostic devices are physical4/5; the historical formal incident used physical2/3.',
            'No forward/backward/operator cause was reproduced; passing this fresh chain is not a repair.',
            'Recorded state equality only covers listed flags/FSDP state, not every possible internal value.',
            'Anomaly detection and synchronous observations add overhead; duration is not formal training throughput.',
        ])
    (ROOT/'experiments/rl/results_dt_to_native_actor_nonfinite_20261008.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','root_cause_located','fixed','full_fresh_update_finite','phase_seconds','maximum_observed_tree_pss_bytes','maximum_observed_physical_GPU_MiB','remaining_owned_non_zombie_processes')},indent=2))


if __name__ == '__main__':
    main()
