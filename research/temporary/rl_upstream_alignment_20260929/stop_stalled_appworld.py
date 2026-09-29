"""Stop only the recorded AppWorld pilot after capturing its DataLoader stall."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil

audit = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/upstream-alignment-20260929')
job = next(j for j in json.loads((audit/'two-gpu-pilots/manifest.json').read_text())
           if j['method'] == 'dt' and j['task'] == 'AppWorld')
assert job['pid'] == 210974 and os.getsid(job['pid']) == job['pid']
parent = psutil.Process(job['pid'])
assert 'run_two_gpu_training_pilot.sh' in ' '.join(parent.cmdline())
directory = Path(job['directory'])
assert (directory/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == '1'
processes = [parent] + parent.children(recursive=True)
assert {211729, 222430, 227927, 230650, 406057}.issubset({p.pid for p in processes})
stacks = {str(pid): subprocess.run(['/opt/conda/bin/py-spy', 'dump', '-p', str(pid)],
    capture_output=True, text=True, check=True).stdout for pid in (222430, 406057, 227927)}
assert 'torchdata/stateful_dataloader/worker.py:99' in stacks['406057']
assert '_reset (torchdata/stateful_dataloader/stateful_dataloader.py:' in stacks['222430']
record = dict(stopped_unix=time.time(), reason='Next epoch DataLoader fork child stalled in torch.set_num_threads(1); policy workers idle',
              checkpoint_marker=1, process_ids=[p.pid for p in processes], stacks=stacks)
(directory/'dataloader-stall-stop.json').write_text(json.dumps(record, indent=2)+'\n')
# Each Process object retains its creation identity. No broad Ray/service kill.
for process in reversed(processes):
    try:
        process.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        pass
_, alive = psutil.wait_procs(processes, timeout=8)
for process in alive:
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
record['remaining_pids'] = [p.pid for p in psutil.wait_procs(alive, timeout=4)[1]]
(directory/'dataloader-stall-stop.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k != 'stacks'}))
