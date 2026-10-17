"""Sequential process composition around the original fixed VERL trainer.

VERL owns sampling, advantages, updates, stopping, saving and restoration.
This file starts the next job only after native success and a complete budget.
It does not skip failed jobs or implement a second training/retry loop.
"""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import time

import psutil
from prepare_campaign import ROOT, AUDIT, VERSION, sha


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def publish_job(row):
    for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
        path = ROOT / name
        value = json.loads(path.read_bytes())
        value['jobs'] = [j for j in value['jobs'] if not
                         (j.get('task') == row['task'] and j.get('method') == row['method'])] + [row]
        value.update(updated_unix=time.time(), current_sequential_campaign=str(AUDIT / 'campaign-state.json'))
        write(path, value)


def original_log_reader():
    path = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
    assert sha(path) == 'c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
    return runpy.run_path(str(path), run_name='original_eight_gpu_campaign_log_reader')


def release_finished_session(pid, birth):
    """Use the prior stop owner's exact process block for this finished session."""
    members = []
    for p in psutil.process_iter(['pid', 'create_time']):
        try:
            if os.getsid(p.pid) == pid and p.create_time() >= birth and p.status() != psutil.STATUS_ZOMBIE:
                members.append(p)
        except (psutil.Error, OSError):
            pass
    if not members:
        return dict(remaining_non_zombie=[], session_members=0)
    owner = AUDIT / 'source/resume_at_native_checkpoint.py'
    assert sha(owner) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
    template = next(n.value for n in ast.walk(ast.parse(owner.read_text()))
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and 'processes=[parent]+parent.children(recursive=True)' in n.value)
    block = template[template.index('for p in reversed(processes):'):template.index("record['other_jobs_unchanged']={}")]
    record = dict(session_members=len(members), session_id=pid, driver_birth=birth)
    exec(compile(block, str(owner), 'exec'), dict(processes=members, record=record,
                                               psutil=psutil, signal=signal, time=time))
    return record


def verify_GPU_empty():
    physical = subprocess.check_output(['mx-smi'], text=True)
    table = physical.split('| Process:')[-1]
    for device in range(8):
        assert not re.search(r'\|\s*' + str(device) + r'\s+\d+\s+', table), 'Physical GPU occupied'
    return physical


def main():
    with (AUDIT / 'campaign-controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = json.loads((AUDIT / 'campaign-prepared.json').read_bytes())
        assert prepared['status'] == 'three_native_CPU_configs_resolved_and_common_workload_equal'
        stopped = json.loads((AUDIT / 'user-replaced-prior-jobs-stop.json').read_bytes())
        assert stopped['status'] == 'stopped_by_user_replacement' and not stopped['remaining_non_zombie']
        assert not (AUDIT / 'campaign-state.json').exists(), 'Inspect existing campaign identity; no duplicate submission'
        state = dict(version=VERSION, status='starting', started_unix=time.time(),
                     controller_pid=os.getpid(), controller_birth=psutil.Process().create_time(),
                     source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
                     order=prepared['order'], jobs=[], current_method=None)
        owner = original_log_reader()
        write(AUDIT / 'campaign-state.json', state)
        try:
            for method in state['order']:
                row = json.loads((AUDIT / (method + '-prepared.json')).read_bytes())
                assert all(sha(p) == h for p, h in row['source_bindings'].items())
                physical = verify_GPU_empty()
                run, output = Path(row['run']), Path(row['output'])
                run.mkdir(parents=True, exist_ok=False)
                output.mkdir()
                checkpoint = Path(row['checkpoints'])
                checkpoint.mkdir(parents=True, exist_ok=False)
                env = dict(os.environ)
                for key in ('DT_NATIVE_HOST_MEMORY_LEASE', 'MACA_VISIBLE_DEVICES', 'RAY_ADDRESS', 'RAY_TMPDIR'):
                    env.pop(key, None)
                env.update(row['effective_environment'])
                assert 'DT_NATIVE_HOST_MEMORY_LEASE' not in env
                stdout, stderr = output / 'train.log', run / 'driver.stderr.log'
                (run / 'physical-GPU-before-launch.txt').write_text(physical)
                with stdout.open('xb') as out, stderr.open('xb') as err:
                    child = subprocess.Popen(row['argv'], cwd=row['entry'], env=env,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                row.update(pid=child.pid, birth=psutil.Process(child.pid).create_time(),
                    started_unix=time.time(), observed_process_created_unix=psutil.Process(child.pid).create_time(),
                    source_receipt=str(run / 'source.json'), log=str(stdout), stdout=str(stdout), stderr=str(stderr),
                    exit_file=str(run / 'exit-code.txt'), active=True, alive=True,
                    local_patch_commit=os.environ['RL_CAMPAIGN_COMMIT'],
                    status='submitted_initialization_pending', initialization_is_not_training_progress=True)
                write(run / 'launch.json', row)
                write(run / 'source.json', row)
                publish_job(row)
                state.update(status='running_native_job', current_method=method)
                state['jobs'].append(row)
                write(AUDIT / 'campaign-state.json', state)
                while child.poll() is None:
                    # Existing collector resolves this driver's Ray session and
                    # reads the original TaskRunner logs. No training RPC runs.
                    try:
                        snapshot = owner['collect'](str(ROOT), [row['output']])
                        current, = snapshot['jobs']
                        if current.get('ray_session'):
                            row['ray_session'] = current['ray_session']
                        row.update(completed_iterations=owner['completed'](current),
                                   phase=current.get('phase_label'), progress_collected=snapshot['collected_at'])
                        write(AUDIT / (method + '-latest-source-snapshot.json'), snapshot)
                    except Exception as exc:
                        # Optional observation must not terminate a valid native
                        # training process or masquerade as completed progress.
                        row['optional_observation_unavailable'] = dict(error_type=type(exc).__name__,
                                                                     collected=time.time())
                    publish_job(row)
                    write(run / 'source.json', row)
                    write(AUDIT / 'campaign-state.json', state)
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
                Path(row['exit_file']).write_text(str(child.returncode) + '\n')
                snapshot = owner['collect'](str(ROOT), [row['output']])
                current, = snapshot['jobs']
                write(AUDIT / (method + '-final-source-snapshot.json'), snapshot)
                assert child.returncode == 0, 'Native job failed; retain this combination and do not skip it'
                assert owner['completed'](current) == 200
                assert {m['step'] for m in current['metrics']} >= set(range(1, 201)), 'Original complete-iteration metrics missing'
                assert current['checkpoint_step'] == 200, 'Native final checkpoint not completed'
                final = checkpoint / 'global_step_200'
                roles = ('actor', 'critic') if method == 'ppo' else ('actor',)
                files = [final / 'data.pt'] + [final / role / f'{kind}_world_size_8_rank_{rank}.pt'
                    for role in roles for rank in range(8) for kind in ('model', 'optim', 'extra_state')]
                assert all(p.is_file() and p.stat().st_size for p in files), 'Native final state files missing'
                row.update(status='completed_200_native_iterations', active=False, alive=False,
                           completed_iterations=200, final_native_checkpoint=str(final), finished_unix=time.time(),
                           final_checkpoint_files={str(p.relative_to(final)):p.stat().st_size for p in files})
                row['native_finished_session_cleanup'] = release_finished_session(row['pid'], row['birth'])
                assert not row['native_finished_session_cleanup']['remaining_non_zombie']
                write(run / 'source.json', row)
                publish_job(row)
                write(AUDIT / 'campaign-state.json', state)
            state.update(status='all_three_native_200_iteration_runs_completed', finished_unix=time.time(), current_method=None)
            write(AUDIT / 'campaign-state.json', state)
        except BaseException as exc:
            state.update(status='needs_attention_no_failed_combination_skipped', error_type=type(exc).__name__,
                         error=str(exc), observed_unix=time.time())
            write(AUDIT / 'campaign-state.json', state)
            raise


if __name__ == '__main__':
    main()
