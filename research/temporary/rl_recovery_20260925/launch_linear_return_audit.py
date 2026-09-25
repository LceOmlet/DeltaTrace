"""Replay saved first-rollout IDs through native owners with passive diagnostics."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
rec = root/'receipts/rollout-major-cost'
release = root/'releases/64e5876'
source = rec/'linear-return-first-minimum.json'
name = 'linear-return-boundary-audit'
assert source.exists() and not (rec/(name+'-job.json')).exists()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 29756))
script = rec/(name+'.sh')
script.write_text(f'''#!/bin/bash
export DT_ROOT={release}
export DT_ENVIRONMENT_JSON="$DT_ROOT/environment.json"
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export CUDA_VISIBLE_DEVICES=7 MACA_VISIBLE_DEVICES=7 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29756
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="{rec}:$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT:${{PYTHONPATH:-}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 PYTHONHASHSEED=0
export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
export CREDIT_ROUTE_SOURCE={source}
export CREDIT_ROUTE_OUTPUT={rec}/{name}.json
export CREDIT_ROUTE_AUDIT=1 CREDIT_ROUTE_AUDIT_LAYERS=2,3,4,5
"$VENV_PYTHON" -u "{rec}/compare_credit_routes.py"
rc=$?
printf '%s\\n' "$rc" > "{rec}/{name}.exitcode"
exit "$rc"
''')
subprocess.run(['bash', '-n', str(script)], check=True)
with (rec/(name+'.log')).open('w') as log:
    proc = subprocess.Popen(['bash', str(script)], stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
job = dict(pid=proc.pid, started=time.time(), gpu=7, script=str(script),
    source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    scope='Saved first pilot DT batch before any actor update; fresh initial actor. '
          'Original native forward and formal DT only; passive boundary contractions, no numerical gate or method change.')
(rec/(name+'-job.json')).write_text(json.dumps(job, indent=2)+'\n')
print(json.dumps(job))
