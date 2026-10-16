"""Stop the exact GPU6/7 inference service explicitly named by the user."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-sql-native-speed-20261016-v2'
AUDIT.mkdir(exist_ok=True)
receipt = AUDIT / 'GPU67-user-stopped.json'
assert not receipt.exists(), 'Read the existing stop receipt before repeating signals'
driver = psutil.Process(2614045)
assert driver.create_time() == 1791949034.31
assert driver.cmdline()[1:3] == ['-m', 'vllm.entrypoints.openai.api_server']
assert os.getsid(driver.pid) == driver.pid == os.getpgid(driver.pid)
for pid in (2617231, 2617232):
    process = psutil.Process(pid)
    assert process.create_time() == 1791949058.07
    assert os.getpgid(pid) == driver.pid

before = subprocess.check_output(['mx-smi'], text=True)
gpu_rows = []
for line in before.splitlines():
    match = re.match(r'\|\s*(\d+)\s+(\d+)\s+\S+', line)
    if match:
        gpu_rows.append((int(match[1]), int(match[2])))
assert {(gpu, pid) for gpu, pid in gpu_rows if gpu in (6, 7)} == {(6, 2617231), (7, 2617232)}
protected = set(pid for gpu, pid in gpu_rows if gpu not in (6, 7))
protected_jobs = []
for pid, birth in ((878971, 1792091830.03), (3946004, 1792041120.19)):
    process = psutil.Process(pid)
    assert process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE
    protected_jobs.append(dict(pid=pid, birth=birth))
    protected.add(pid)
    protected.update(child.pid for child in process.children(recursive=True))

members = {driver.pid: driver}
members.update({child.pid: child for child in driver.children(recursive=True)})
for process in psutil.process_iter():
    try:
        if os.getpgid(process.pid) == driver.pid:
            members[process.pid] = process
    except (ProcessLookupError, psutil.NoSuchProcess):
        pass
assert not protected.intersection(members)
identities = [dict(pid=p.pid, birth=p.create_time(), name=p.name()) for p in members.values()]
report = dict(started_unix=time.time(), instruction='User requested stopping physical ID6/7 and continuing SQL/AppWorld efficiency trials',
              driver=dict(pid=driver.pid, birth=driver.create_time()), physical_gpus=[6, 7],
              members_before=identities, protected_jobs=protected_jobs, mx_smi_before=before,
              signals=[], files_deleted=False, new_checkpoints=False, status='validated_before_stop')
receipt.write_text(json.dumps(report, indent=2) + '\n')
os.killpg(driver.pid, signal.SIGTERM)
report['signals'].append(dict(signal='SIGTERM', process_group=driver.pid, unix=time.time()))
_, survivors = psutil.wait_procs(list(members.values()), timeout=12)
for process in survivors:
    old = next(item for item in identities if item['pid'] == process.pid)
    try:
        if process.create_time() == old['birth'] and process.status() != psutil.STATUS_ZOMBIE:
            assert process.pid not in protected
            process.kill()
            report['signals'].append(dict(signal='SIGKILL', pid=process.pid, birth=old['birth']))
    except psutil.NoSuchProcess:
        pass
psutil.wait_procs(survivors, timeout=5)
report['survivors'] = []
for old in identities:
    try:
        process = psutil.Process(old['pid'])
        if process.create_time() == old['birth'] and process.status() != psutil.STATUS_ZOMBIE:
            report['survivors'].append(old)
    except psutil.NoSuchProcess:
        pass
assert not report['survivors']
for old in protected_jobs:
    process = psutil.Process(old['pid'])
    assert process.create_time() == old['birth'] and process.status() != psutil.STATUS_ZOMBIE
report.update(status='stopped_at_user_request', finished_unix=time.time(),
              mx_smi_after=subprocess.check_output(['mx-smi'], text=True))
receipt.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(receipt), status=report['status'], survivors=report['survivors'],
                     protected_jobs=protected_jobs, mx_smi_after=report['mx_smi_after'])))
