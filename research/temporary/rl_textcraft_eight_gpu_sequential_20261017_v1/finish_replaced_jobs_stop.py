"""Reconcile delayed native GPU teardown without sending any further signal."""
import json
import os
from pathlib import Path
import time
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1'
path = AUDIT / 'user-replaced-prior-jobs-stop.json'
record = json.loads(path.read_bytes())
preserved = AUDIT / 'user-replaced-prior-jobs-native-stop-first-observation.json'
if not preserved.exists():
    preserved.write_bytes(path.read_bytes())
remaining = []
for row in record['processes']:
    try:
        p = psutil.Process(row['pid'])
        if p.create_time() == row['birth'] and p.status() != psutil.STATUS_ZOMBIE:
            remaining.append(row)
    except psutil.NoSuchProcess:
        pass
assert not remaining, 'Original identities still releasing resources; no additional signals or launch'
record.update(status='stopped_by_user_replacement', remaining_non_zombie=[],
              finished_unix=time.time(), first_native_stop_observation=str(preserved),
              delayed_GPU_teardown_reconciled=True, additional_signals=0)
path.write_text(json.dumps(record, indent=2) + '\n')
identities = {(r['pid'], r['birth']) for r in record['roots']}
for name in ('formal-training.json', 'active-training.json', 'active-source.json',
             'receipts/appworld-formal-training-20261017-v1/formal-training.json'):
    p = ROOT / name
    value = json.loads(p.read_bytes())
    for job in value['jobs']:
        if (job.get('pid'), job.get('birth')) in identities:
            job.update(status='stopped_by_user_for_fresh_eight_GPU_comparison', active=False, alive=False,
                       stopped_unix=record['finished_unix'], stop_receipt=str(path))
    tmp = p.with_name(p.name + '.tmp-eight-gpu')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, p)
print(json.dumps(dict(status=record['status'], remaining=[], additional_signals=0, receipt=str(path))))
