"""Record the native failure and correct stale liveness/observation claims.

This administrative action reads the original logs and process identities. It
does not signal a process, submit training, or alter any training configuration.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
RUN = ROOT / 'runs/appworld-dtpo-reward-target-exponential-variance-formal-20261017-v6'
OUTPUT = RUN / 'runtime-overrides/native-driver-exit-after-manual-worker-suspension-20261017-v1.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    launch = json.loads((RUN / 'launch.json').read_bytes())
    identities = [(3805315, 1792200560.89), (3809067, 1792200587.19),
                  (3815384, 1792200610.96), (3817124, 1792200628.36)]
    processes = []
    for pid, birth in identities:
        try:
            process = psutil.Process(pid)
            alive = process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        processes.append(dict(pid=pid, birth=birth, alive=alive))
    assert not any(p['alive'] for p in processes)
    stderr_path = Path(launch['stderr'])
    stderr = stderr_path.read_text(errors='replace').splitlines()
    matches = [i for i, line in enumerate(stderr) if 'keepalive watchdog timeout' in line]
    assert len(matches) == 1
    index = matches[0]
    native_error = [dict(path=str(stderr_path), line=i + 1, text=stderr[i])
                    for i in range(max(0, index - 18), min(len(stderr), index + 2))]
    suspension = json.loads((AUDIT / 'AppWorld-DT2-temporary-resource-suspension-v1.json').read_bytes())
    priorities = json.loads((AUDIT / 'native-sibling-OOM-protection-during-AppWorld-DT2-v1.json').read_bytes())
    assert priorities['status'] == 'temporary_sibling_priority_restored_after_DT_workers_exited'
    assert not priorities.get('failures')
    marker = Path(launch['checkpoints']) / 'latest_checkpointed_iteration.txt'
    assert not marker.exists(), 'Inspect a newly completed checkpoint before changing the recorded claim'
    latest_resources = json.loads((ROOT / 'receipts/appworld-formal-training-20261017-v1/latest-host-RSS-PSS-XTT-v1.json').read_bytes())
    global_oom = int(next(l.split()[1] for l in Path('/proc/vmstat').read_text().splitlines()
                          if l.startswith('oom_kill ')))
    record = dict(
        status='native_driver_failed_after_manual_worker_suspension', collected=time.time(),
        task='AppWorld', method='dt', method_version='v6', run=str(RUN),
        identities=processes, completed_formal_iterations=1, total_iterations=200,
        interrupted_iteration=2, completed_DT_B4_per_rank=25, total_DT_B4_per_rank=27,
        native_full_checkpoint=None, native_checkpoint_marker=str(marker),
        lost_unsaved_progress='One completed iteration and the partial second iteration; no checkpoint exists to resume them',
        native_exception='RayTaskError(ActorUnavailableError): RPC keepalive watchdog timeout',
        original_driver_error=native_error,
        driver_stderr_SHA256=hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
        causal_intervention='Manual SIGSTOP held both GPU workers; the RPC keepalive watchdog expired before SIGCONT. This intervention was erroneous.',
        suspension_receipt=str(AUDIT / 'AppWorld-DT2-temporary-resource-suspension-v1.json'),
        elapsed_worker_suspension_seconds=suspension['elapsed_resource_hold_seconds'],
        OOM_at_failure=dict(global_oom_kill=global_oom,
                            cgroup_oom_control=latest_resources['cgroup_oom_control'],
                            note='No counter increase from four; this exit is not recorded as a new kernel OOM'),
        sibling_OOM_priorities_restored_receipt=str(AUDIT / 'native-sibling-OOM-protection-during-AppWorld-DT2-v1.json'),
        all_319_original_sibling_OOM_priorities_restored=True,
        training_signals_from_this_recorder=0, model_calls=0, optimizer_updates=0,
        training_configuration_changes=0, goal_status='active',
        next_submission='Not submitted; native 32k four-job host-memory safety remains unverified. Preserve failed v6 history and use a distinct execution identity for any fresh retry.')
    write(OUTPUT, record)
    for path in [ROOT / 'formal-training.json', ROOT / 'active-training.json', ROOT / 'active-source.json',
                 ROOT / 'receipts/appworld-formal-training-20261017-v1/formal-training.json']:
        data = json.loads(path.read_bytes())
        before_path = AUDIT / ('AppWorld-DT-suspension-failure-before-' + ('campaign-' if path.parent.name == 'appworld-formal-training-20261017-v1' else '') + path.name)
        if not before_path.exists():
            write(before_path, deepcopy(data))
        job, = [j for j in data['jobs'] if j.get('task') == 'AppWorld' and j.get('method') == 'dt']
        assert job['pid'] == launch['pid'] and job['birth'] == launch['birth']
        job.update(alive=False, active=False, status=record['status'],
                   failure_receipt=str(OUTPUT), completed_formal_iterations=1,
                   latest_completed_native_checkpoint=None)
        data['updated_unix'] = time.time()
        write(path, data)
    previous = AUDIT / 'read-only-native-phase-wait-job.json'
    if previous.exists():
        old = json.loads(previous.read_bytes())
        before = AUDIT / 'read-only-native-phase-wait-job-before-failure-correction.json'
        if not before.exists():
            write(before, old)
        old.update(status='observation_finished_native_DT_RPC_failed',
                   previous_training_status=old.get('training_status'),
                   training_status='native_second_DT_not_completed_RPC_keepalive_watchdog_timeout',
                   failure_receipt=str(OUTPUT), corrected_unix=time.time(),
                   note='A finally-block lease release is not evidence of successful DT completion.')
        write(previous, old)
    print(json.dumps({k: record[k] for k in ('status', 'completed_formal_iterations',
                                           'native_full_checkpoint', 'native_exception', 'goal_status')}))


if __name__ == '__main__':
    main()
