"""Read-only wait for the same live PPO's next native completed checkpoint.

No signal, training call, pause or diagnostic submission is performed here.
"""
import json
import time
from pathlib import Path

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
job = json.loads((OUT / 'native-PPO-control-v9-job.json').read_bytes())
assert (job['pid'], job['birth']) == (1308636, 1792016095.73)
marker = Path(job['run']) / 'textcraft-ppo/checkpoints/latest_checkpointed_iteration.txt'
path = OUT / 'verified-PPO-native-checkpoint-wait-v82.json'
last_print = 0
while True:
    p = psutil.Process(job['pid'])
    assert p.create_time() == job['birth'] and p.status() != psutil.STATUS_ZOMBIE
    current = json.loads((OUT / 'native-PPO-control-v9-job.json').read_bytes())
    assert (current['pid'], current['birth']) == (job['pid'], job['birth'])
    step = int(marker.read_text())
    now = time.time()
    report = dict(unix=now, pid=job['pid'], birth=job['birth'], marker=str(marker),
                  completed_checkpoint=step, required_checkpoint=30,
                  status='completed_native_checkpoint_available' if step >= 30 else 'PPO_still_running_native_checkpoint_wait',
                  workers=[dict(pid=c.pid, birth=c.create_time(), phase=c.name())
                           for c in p.children(recursive=True) if c.name().startswith('ray::WorkerDict')],
                  model_backward_optimizer_scheduler_calls=0, signals_sent=0,
                  hourly_automation='PAUSED')
    path.write_text(json.dumps(report, indent=2) + '\n')
    if now - last_print >= 60 or step >= 30:
        print(json.dumps(report), flush=True)
        last_print = now
    if step >= 30:
        break
    time.sleep(20)
