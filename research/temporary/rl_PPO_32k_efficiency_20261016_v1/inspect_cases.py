"""Read bounded case status/log tails and physical GPU samples; no model RPC."""
import json
from pathlib import Path
import subprocess
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1'
for name in ('baseline-fit','candidate-fit','reference-32k','candidate-32k'):
    folder=AUDIT/name
    if not (folder/'launched.json').exists():continue
    launched=json.loads((folder/'launched.json').read_bytes())
    try:
        process=psutil.Process(launched['pid'])
        live=process.create_time()==launched['birth'] and process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:live=False
    state=json.loads((folder/'status.json').read_bytes()) if (folder/'status.json').exists() else None
    print(json.dumps(dict(case=name,launcher_live=live,state=state),ensure_ascii=False))
    if (folder/'driver.log').exists():
        with (folder/'driver.log').open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-6500))
            print(f.read().decode(errors='replace'))
print(subprocess.check_output(['mx-smi'],text=True))
