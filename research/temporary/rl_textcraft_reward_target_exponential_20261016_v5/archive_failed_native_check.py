"""Preserve the failed harness import attempt before the corrected check."""
import json
from pathlib import Path

import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
job=json.loads((AUDIT/'native-behavior-job.json').read_bytes())
state=json.loads((AUDIT/'native-behavior-status.json').read_bytes())
assert state['status']=='failed'
assert not psutil.pid_exists(job['pid']) or psutil.Process(job['pid']).create_time()!=job['birth']
destination=AUDIT/'failed-harness-import-r1';destination.mkdir()
for name in ('native-behavior-job.json','native-behavior-status.json','native-behavior.stdout.log','native-behavior.stderr.log'):
    (AUDIT/name).rename(destination/name)
print('Preserved failed Ray test-harness import and terminal status; no model/optimizer or formal job changed.')
