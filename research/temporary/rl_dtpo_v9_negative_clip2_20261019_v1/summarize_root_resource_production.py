"""Read the first complete production window after the root-storage override.

The pinned owner still parses dmon, aligns adjacent host observations and reads
training logs. This adapter only selects the deployed run and time window.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
SEGMENT = ROOT / 'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v4'
RUN = ROOT / 'runs/textcraft-dtpo-v9-negative-clip2-8gpu-200-20261019-v4'


def main():
    override_path = RUN / 'runtime-overrides/root-pageable-resource-override-completed.json'
    override = json.loads(override_path.read_bytes())
    assert override['status'] == 'completed_runtime_root_storage_override'
    assert len(override['actual_workers']) == 8
    lower = max(r['override']['applied_unix'] for r in override['actual_workers'])
    reader = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
    assert hashlib.sha256(reader.read_bytes()).hexdigest() == 'c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
    owner = runpy.run_path(str(reader), run_name='original_root_resource_progress_reader')
    state = json.loads((SEGMENT / 'campaign-state.json').read_bytes())
    row = state['jobs'][0]
    assert row['run'] == str(RUN)
    snapshot = owner['collect'](str(ROOT), [row['output']])
    job, = snapshot['jobs']
    assert job['pid_alive'], 'Current native driver identity did not match'
    assert owner['completed'](job) >= 14, 'First production iteration is not complete yet'
    upper = snapshot['collected_at']

    source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/summarize_native_utilization_v2.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e6f3b17198f34ca0b5946637dcbd3ee4920bb08b9783bbc374ffe85bb14199e3'
    destination = AUDIT / ('root-resource-production-' + str(int(upper)))
    destination.mkdir(exist_ok=False)
    code = source.read_text()
    code = code.replace("AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'",
                        'AUDIT = Path(' + repr(str(destination)) + ')')
    code = code.replace("PHYSICAL = AUDIT / 'native-mx-smi-dmon-timestamped.jsonl'",
                        "PHYSICAL = ROOT / 'receipts/six-run-native-utilization-20261017-v1/native-mx-smi-dmon-timestamped.jsonl'")
    code = code.replace('-8gpu-200-20261017-v1', '-8gpu-200-20261019-v4')
    assert code.count('lower, upper = None, None') == 1
    code = code.replace('lower, upper = None, None', f'lower, upper = {lower!r}, {upper!r}')
    namespace = {'__name__': 'original_root_resource_production_reducer', '__file__': str(source)}
    exec(compile(code, str(source), 'exec'), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        namespace['main']()
    physical = json.loads((destination / 'native-utilization-phase-summary.json').read_bytes())
    (destination / 'source-snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    groups = []
    for phase in physical['phases']:
        devices = list(phase['devices'].values())
        n = sum(d['gpu_utilization_percent']['samples'] for d in devices)
        groups.append(dict(identity=phase['identity'], card_samples=n,
                           mean_GPU_Util_percent=sum(d['gpu_utilization_percent']['mean'] *
                                                     d['gpu_utilization_percent']['samples']
                                                     for d in devices) / n))
    checkpoint = Path(row['checkpoints'])
    step = int((checkpoint / 'latest_checkpointed_iteration.txt').read_text())
    final = checkpoint / ('global_step_' + str(step))
    files = [final / 'data.pt'] + [final / 'actor' / f'{kind}_world_size_8_rank_{rank}.pt'
             for rank in range(8) for kind in ('model', 'optim', 'extra_state')]
    checkpoint_record = dict(step=step, directory=str(final),
        criterion_owner=str(ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'),
        files=[dict(path=str(p), bytes=p.stat().st_size if p.is_file() else None) for p in files],
        all_present_nonempty=all(p.is_file() and p.stat().st_size > 0 for p in files))
    # These are observed instrumentation scopes in this fixed run, not method
    # exclusions. Reward/learning curves retain their original iterations.
    excluded_validation_steps = [7, 8, 11, 13]
    if (AUDIT / 'native-DT-replay-trace-receipt.json').exists():
        excluded_validation_steps.append(19)
    resident_pair = AUDIT / 'existing-DT-gpu-resident-pair-results.json'
    if resident_pair.exists():
        pair = json.loads(resident_pair.read_bytes())
        if pair.get('additional_calls', {}).get('DT', 0):
            excluded_validation_steps.append(21)
    for validation_name in ('existing-DT-FA-capture-resource-pair-results.json',
                            'existing-DT-long-GPU-resident-resource-pair-results.json'):
        validation_path = AUDIT / validation_name
        if validation_path.exists():
            validation = json.loads(validation_path.read_bytes())
            excluded_validation_steps.extend(validation.get('training_iterations_excluded_from_production_efficiency', []))
    excluded_validation_steps = sorted(set(excluded_validation_steps))
    report = dict(status='completed_read_only_production_window', collected=time.time(),
        run=str(RUN), source_snapshot=str(destination / 'source-snapshot.json'),
        actual_execution_source=row['local_patch_commit'], method_source=override['method_commit'],
        override_receipt=str(override_path), effective_flags=override['effective_root_flags'],
        original_reader_SHA256=hashlib.sha256(reader.read_bytes()).hexdigest(),
        physical_reducer_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        physical_report=str(destination / 'native-utilization-phase-summary.json'),
        window=physical['window'], groups=groups,
        latest_completed=owner['completed'](job), current_phase=job['phase_label'],
        all_metrics=job['metrics'], production_steps_after_override=[m for m in job['metrics']
            if m['step'] >= 14 and m['step'] not in excluded_validation_steps],
        excluded_validation_steps=excluded_validation_steps, passively_observed_steps=[17, 18], checkpoint=checkpoint_record,
        model_calls=0, training_mutations=0,
        interpretation='Native device activity, not peak FLOPS. Mixed/missing brackets remain separate. Different input lengths prevent a single-iteration causal speedup claim. The cumulative physical window can include the separately recorded profiled/validation intervals; it is not an uninstrumented throughput comparison. production_steps_after_override excludes known model-validation/profile iterations, while reward curves retain all original updates.')
    target = destination / 'production-window-receipt.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(target), latest_completed=report['latest_completed'],
                          current_phase=report['current_phase'], groups=groups,
                          checkpoint_step=step, checkpoint_complete=checkpoint_record['all_present_nonempty'])))


if __name__ == '__main__':
    main()
