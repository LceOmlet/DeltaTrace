"""Timestamp the native mx-smi dmon stream; never call a training API.

The installed dmon CSV omits timestamps. Preserve each original line and its
receive time so it can be joined to the existing native process-phase record.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'


def main():
    job_path = AUDIT / 'physical-observer-job.json'
    assert not job_path.exists(), 'Inspect the previous observation identity before another launch'
    argv = ['mx-smi', 'dmon', '--show-usage', '--show-memory', '--show-board-power',
            '--format', 'csv', '-l', '2000', '-c', '0']
    record = dict(pid=os.getpid(), birth=psutil.Process().create_time(), started=time.time(),
        source_commit=os.environ['RL_UTILIZATION_COMMIT'], source_path=__file__,
        source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), native_argv=argv,
        status='starting_native_physical_observation', model_calls=0, training_changes=0,
        phase_source=str(ROOT / 'receipts/appworld-formal-training-20261017-v1/native-host-RSS-PSS-XTT-timeseries-v1.jsonl'),
        phase_alignment='Original worker process titles sampled every10 seconds; boundary uncertainty retained',
        physical_scope='Original dmon utilization/memory/power counters; receive timestamps, not GPU kernel timing')
    with (AUDIT / 'native-mx-smi-dmon.csv').open('x') as raw, \
            (AUDIT / 'native-mx-smi-dmon-timestamped.jsonl').open('x') as timed, \
            (AUDIT / 'native-mx-smi-dmon.stderr').open('x') as errors:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=errors, text=True, bufsize=1)
        record.update(status='recording_native_physical_utilization', native_pid=child.pid,
                      native_birth=psutil.Process(child.pid).create_time())
        job_path.write_text(json.dumps(record, indent=2) + '\n')
        last_completion_check = 0.0
        try:
            for line in child.stdout:
                now = time.time()
                raw.write(line); raw.flush()
                timed.write(json.dumps(dict(received_unix=now, native_csv_line=line.rstrip('\n'))) + '\n')
                timed.flush()
                if now - last_completion_check >= 60:
                    last_completion_check = now
                    paths = [ROOT / 'receipts' / (task + '-eight-gpu-sequential-20261017-v1/campaign-state.json')
                             for task in ('textcraft', 'appworld')]
                    if all(p.exists() and json.loads(p.read_bytes())['status'] ==
                           'all_three_native_200_iteration_runs_completed' for p in paths):
                        record['status'] = 'all_six_native_runs_completed_physical_observation_finished'
                        break
        finally:
            if child.poll() is None:
                child.terminate()
            record.update(native_returncode=child.wait(timeout=10), finished=time.time())
            if record['status'] == 'recording_native_physical_utilization':
                record['status'] = 'native_physical_observation_stream_ended'
            job_path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
