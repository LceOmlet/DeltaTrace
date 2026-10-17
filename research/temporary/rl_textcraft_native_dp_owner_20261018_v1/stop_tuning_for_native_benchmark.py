"""Bind the original protected stop owner to the identified tuning job only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-native-dp-owner-20261018-v1'
CURRENT = ROOT / 'receipts/six-run-fresh-after-register-threads-20261017-v1'
SEGMENT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v3'
IDENTITIES = [(453157, 1792249756.70), (453162, 1792249757.08)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    receipt = AUDIT / 'tuning-stopped-for-native-benchmark.json'
    assert not (AUDIT / 'native-benchmark/job.json').exists()
    assert json.loads((AUDIT / 'native-benchmark/CPU-preflight.json').read_bytes())['status'] == 'CPU_configuration_preflight_passed_not_GPU_validation'
    assert json.loads((AUDIT / 'receipts/cpu-v2-result.json').read_bytes())['tests']['failures'] == '0'
    staged = json.loads((AUDIT / 'receipts/remote-candidate-prepared-v2.json').read_bytes())
    assert sha(Path(staged['candidate'])/'verl-owner/verl/workers/fsdp_workers.py') == staged['worker_SHA256']
    owner = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/resume_at_native_checkpoint.py'
    assert sha(owner) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
    if not receipt.exists():
        processes, seen = [], set()
        for pid, birth in IDENTITIES:
            p = psutil.Process(pid)
            assert p.create_time() == birth and p.status() != psutil.STATUS_ZOMBIE
            for q in [p, *p.children(recursive=True)]:
                if q.pid not in seen:
                    seen.add(q.pid)
                    processes.append(q)
        for p in psutil.process_iter(['pid', 'create_time']):
            try:
                if os.getsid(p.pid) == IDENTITIES[1][0] and p.create_time() >= IDENTITIES[1][1] and p.pid not in seen:
                    seen.add(p.pid)
                    processes.append(p)
            except (psutil.Error, OSError):
                pass
        for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
            (AUDIT / ('before-'+name)).write_bytes((ROOT/name).read_bytes())
        (AUDIT / 'tuning-final-campaign-state.json').write_bytes((SEGMENT/'campaign-state.json').read_bytes())
        (AUDIT / 'tuning-final-progress.json').write_bytes((CURRENT/'latest-progress.json').read_bytes())
        record = dict(status='stopping_user_replaced_tuning', started=time.time(),
            reason='User requires efficiency fixes followed by all six runs from base/zero; finite native generation benchmark first',
            roots=[dict(pid=p, birth=b) for p, b in IDENTITIES],
            processes=[dict(pid=p.pid, birth=p.create_time(), name=p.name()) for p in processes],
            original_logs_and_checkpoints_preserved=True, new_checkpoints=0, SIGSTOP_calls=0)
        write(AUDIT/'tuning-stopping-for-native-benchmark.json', record)
        controller = psutil.Process(IDENTITIES[0][0])
        assert controller.create_time() == IDENTITIES[0][1]
        controller.terminate()
        controller.wait(timeout=10)
        template = next(n.value for n in ast.walk(ast.parse(owner.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and 'processes=[parent]+parent.children(recursive=True)' in n.value)
        block = template[template.index('for p in reversed(processes):'):template.index("record['other_jobs_unchanged']={}")]
        exec(compile(block, str(owner), 'exec'), dict(processes=processes, record=record, psutil=psutil, signal=signal, time=time))
        record.update(status='stopped_by_user_replacement', stop_owner=str(owner), stop_owner_SHA256=sha(owner))
        write(receipt, record)
    record = json.loads(receipt.read_bytes())
    remaining = []
    for row in record['processes']:
        try:
            p = psutil.Process(row['pid'])
            if p.create_time() == row['birth'] and p.status() != psutil.STATUS_ZOMBIE:
                remaining.append(row)
        except psutil.NoSuchProcess:
            pass
    assert not remaining, 'Original teardown still in progress; no benchmark launch'
    record.update(remaining_non_zombie=[], finished=time.time())
    write(receipt, record)
    physical = subprocess.check_output(['mx-smi'], text=True)
    table = physical.split('| Process:')[-1]
    for device in range(8):
        assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+', table), 'GPU still occupied'
    (AUDIT/'physical-before-native-benchmark.txt').write_text(physical)
    state = json.loads((SEGMENT/'campaign-state.json').read_bytes())
    state.update(status='user_stopped_tuning_history_pending_native_placement_benchmark', stopped=time.time(), stop_receipt=str(receipt))
    for row in state['jobs']:
        row.update(active=False, alive=False, status='user_stopped_tuning_history')
        write(Path(row['run'])/'source.json', row)
    write(SEGMENT/'campaign-state.json', state)
    for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
        value = json.loads((ROOT/name).read_bytes())
        for row in value['jobs']:
            if (row.get('pid'), row.get('birth')) == IDENTITIES[1]:
                row.update(status='user_stopped_tuning_history', active=False, alive=False, stop_receipt=str(receipt))
        write(ROOT/name, value)
    print(json.dumps(dict(status=record['status'], remaining=0, GPU_benchmark_launched=False)))


if __name__ == '__main__':
    main()
