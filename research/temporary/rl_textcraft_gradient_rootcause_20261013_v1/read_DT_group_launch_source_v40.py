"""Read the actual pinned DT launch and source before the authorized two edits."""
import hashlib
import json
import os
from pathlib import Path
import psutil
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
OLD = ROOT / 'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch = json.loads((OLD / 'launch.json').read_bytes())
env = launch['effective_environment']
verl = Path(env['VERL_ROOT'])
entry = Path(env['PYTHONPATH'].split(':')[0])
files = [verl / 'verl/trainer/ppo/ray_trainer.py', verl / 'verl/trainer/ppo/core_algos.py']
files += [Path(p) / 'dt_training_batch.py' for p in env['PYTHONPATH'].split(':') if (Path(p) / 'dt_training_batch.py').is_file()]
sources = {}
for path in files:
    text = path.read_text()
    local = OUT / ('v40-original-' + path.name)
    local.write_text(text)
    sources[str(path)] = dict(SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), copy=str(local))
processes = []
for pid, birth, role in [(3174448, 1791793629.73, 'protected_GRPO'), (2671393, 1791949551.27, 'protected_PPO'),
                         (3172194, 1791793609.09, 'stopped_DT'), (2510247, 1791907554.02, 'stopped_candidate')]:
    alive = psutil.pid_exists(pid) and psutil.Process(pid).create_time() == birth
    processes.append(dict(pid=pid, birth=birth, role=role, alive=alive))
report = dict(unix=time.time(), launch=launch, source=json.loads((OLD/'source.json').read_bytes()),
              sources=sources, processes=processes,
              physical_mx_smi=subprocess.check_output(['mx-smi'], text=True, timeout=30),
              disk=subprocess.check_output(['df', '-h', str(ROOT), '/tmp'], text=True, timeout=30))
dest = OUT/'DT-group-launch-source-v40.json'
dest.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest), report=report)))
