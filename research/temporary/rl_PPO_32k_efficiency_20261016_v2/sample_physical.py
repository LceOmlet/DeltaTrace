"""Bounded raw physical mx-smi and cgroup samples, with the native replay phase."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2'
with (AUDIT/'physical-samples.jsonl').open('x') as output:
    while True:
        state=json.loads((AUDIT/'same-session-status.json').read_bytes())
        record=dict(unix=time.time(),phase=state['status'],physical_mx_smi=subprocess.check_output(['mx-smi'],text=True))
        p=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
        if p.exists():record['cgroup_usage_bytes']=int(p.read_text())
        output.write(json.dumps(record)+'\n');output.flush()
        if state['status'].startswith('completed') or state['status']=='failed':break
        time.sleep(2)
