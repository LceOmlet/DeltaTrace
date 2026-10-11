"""Run the unchanged incident entry, preserving workers for external CPU reads.

Only this supervisor imports the procfs reader. The Ray driver is stopped after
both native replay tasks entered DT, so it cannot tear down failed workers.
Workers are stopped only after their original terminal GEMM report exists.
Every SIGSTOP is paired with SIGCONT in finally, with PID creation-time checks.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import psutil

from capture_cpu_shard import ANCHOR_ROWS, capture


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
JOB = ROOT / 'receipts/textcraft-native-incident37-external-CPU-postmortem-20261011-v1'
SOURCE_SHA = '1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'


def load(path):
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    started = time.monotonic()
    source_path = ROOT / 'runs/textcraft-formal-stable-20261009-v1/source.json'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == SOURCE_SHA
    source = load(source_path)
    original = ROOT / 'receipts/textcraft-native-incident37-competing-weight-capture-20261011-v1'
    binding = load(original / 'binding-check.json')
    entry = original / 'source/capture_native_incident37_GEMM_rows_20261011.py'
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == binding['source_sha256']
    old = ROOT / 'receipts/textcraft-native-incident37-old-ref-DT-20261011-v1/source'
    asynchronous = ROOT / 'receipts/textcraft-native-incident37-async-head-capture-20261011-v1/source'
    exception = ROOT / 'receipts/textcraft-native-incident37-original-exception-capture-20261011-v2/source'
    for path, digest in [(old / 'replay_native_incident37_old_ref_dt_20261011.py', binding['original_source_sha256']),
                         (asynchronous / 'capture_native_incident37_async_head_20261011.py', binding['async_source_sha256']),
                         (exception / 'replay_native_incident37_exception_capture_20261011.py', binding['exception_wrapper_sha256'])]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    mx = subprocess.run(['mx-smi'], capture_output=True, text=True, check=True, timeout=15).stdout
    assert 'no process found' in mx
    assert psutil.virtual_memory().available > 100_000_000_000
    JOB.mkdir(exist_ok=False)
    (JOB / 'selection.json').write_bytes((original / 'selection.json').read_bytes())
    env = os.environ.copy()
    env.update({key: str(value) for key, value in source['environment'].items()})
    env['PYTHONPATH'] = ':'.join([str(original / 'source'), str(asynchronous), str(exception), str(old), source['pythonpath']])
    env['CUDA_VISIBLE_DEVICES'] = '4,5'
    env['TORCH_TRACE'] = str(JOB / 'torch-trace')
    env.pop('RAY_ADDRESS', None)
    assert not any(env.get(key) for key in ['LD_PRELOAD', 'TORCH_CUDA_SANITIZER', 'CUDA_LAUNCH_BLOCKING', 'MACA_LAUNCH_BLOCKING'])
    command = [sys.executable, str(entry), '--source', str(source_path), '--output', str(JOB / 'result')]
    with (JOB / 'driver.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    driver_birth = psutil.Process(child.pid).create_time()
    record = dict(unix=time.time(), supervisor_pid=os.getpid(), supervisor_birth=psutil.Process().create_time(),
        driver=dict(pid=child.pid, birth=driver_birth), command=command, pythonpath=env['PYTHONPATH'],
        source_sha256=SOURCE_SHA, prelaunch_physical_GPU=mx, environment=source['environment'],
        startup_options=source['startup_options'], source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name('capture_cpu_shard.py'), entry]},
        workers=[], signals=[], CPU_captures=[], production_changes=0, formal_restarted=False,
        phase='original_incident_running', in_worker_additional_imports=0)
    def save():
        record['observed_unix'] = time.time()
        (JOB / 'job.json').write_text(json.dumps(record, indent=2) + '\n')
    def signal_exact(pid, birth, value):
        try:
            if abs(psutil.Process(pid).create_time() - birth) >= .001:
                return False
            os.kill(pid, value)
            record['signals'].append(dict(pid=pid, birth=birth, signal=value, unix=time.time()))
            return True
        except (psutil.NoSuchProcess, ProcessLookupError):
            return False
    stopped = []
    try:
        save()
        while time.monotonic() - started < 900:
            ranks = [load(JOB / 'result' / f'rank{rank}.json') for rank in (0, 1)]
            if all(row and row.get('phase') == 'native_saved_DT_batch_begin' for row in ranks):
                record['workers'] = [dict(rank=rank, pid=row['pid'], birth=row['birth']) for rank, row in enumerate(ranks)]
                assert signal_exact(child.pid, driver_birth, signal.SIGSTOP)
                stopped.append((child.pid, driver_birth))
                record['phase'] = 'driver_stopped_after_tasks_dispatched_workers_running'
                save()
                break
            if child.poll() is not None:
                raise RuntimeError(f'Original driver exited before DT: {child.returncode}')
            time.sleep(.1)
        else:
            raise TimeoutError('No native DT dispatch within diagnostic observation budget')
        done = set()
        while time.monotonic() - started < 1200:
            for worker in record['workers']:
                rank = worker['rank']
                if rank in done:
                    continue
                terminal = load(JOB / 'result' / f'rank{rank}-native-GEMM-capture.json')
                if terminal is None:
                    continue
                assert signal_exact(worker['pid'], worker['birth'], signal.SIGSTOP)
                stopped.append((worker['pid'], worker['birth']))
                record.setdefault('original_GEMM_reports', {})[str(rank)] = terminal
                done.add(rank)
                save()
            if len(done) == 2:
                break
            time.sleep(.05)
        if len(done) != 2:
            raise TimeoutError('Did not preserve both original terminal workers')
        for worker in record['workers']:
            wait_started = time.monotonic()
            while psutil.Process(worker['pid']).status() != psutil.STATUS_STOPPED:
                if time.monotonic() - wait_started > 5:
                    raise TimeoutError('SIGSTOP acknowledgement unavailable; do not read a changing CPU allocation')
                time.sleep(.01)
        record['phase'] = 'workers_stopped_after_original_reports_external_CPU_capture'
        save()
        # Import and read saved CPU tensors here only, after original failure.
        import torch
        torch.set_num_threads(2)
        baseline_path = ROOT / 'receipts/textcraft-native-incident37-native-head-live-capture-20261011-v1/result/rank1-native-head-live-microbatch1.pt'
        baseline = torch.load(baseline_path, map_location='cpu', weights_only=False, mmap=True)['tensors']['vocab_weights']
        anchors = {row:baseline[124160 + row, :32].contiguous().view(torch.uint8).numpy().tobytes() for row in ANCHOR_ROWS}
        record['CPU_anchor_sha256'] = {str(row):hashlib.sha256(value).hexdigest() for row,value in anchors.items()}
        for worker in record['workers']:
            if worker['rank'] == 1:
                result = capture(worker['pid'], anchors, JOB / 'external-rank1-CPU')
                record['CPU_captures'].append(result)
                save()
        record['phase'] = 'external_CPU_capture_completed'
    except BaseException:
        record['error'] = traceback.format_exc()
        record['phase'] = 'external_supervisor_error_not_production_repair'
    finally:
        for pid, birth in reversed(stopped):
            signal_exact(pid, birth, signal.SIGCONT)
        save()
        try:
            record['driver_returncode'] = child.wait(timeout=90)
        except subprocess.TimeoutExpired:
            record['driver_still_running'] = True
        record['seconds'] = time.monotonic() - started
        record['supervisor_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        save()


if __name__ == '__main__':
    main()
