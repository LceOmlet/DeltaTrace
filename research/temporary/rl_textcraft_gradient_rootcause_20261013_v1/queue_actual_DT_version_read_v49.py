"""Queue the existing bounded read client; do not interrupt a training RPC."""
import hashlib
import json
import os
from pathlib import Path
import psutil
import subprocess
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
source=OUT/'read_actual_DT_worker_v49.py'
assert not (OUT/'actual-DT-worker-bindings-v49.json').exists(), 'Read the completed result'
for p in psutil.process_iter(['pid','cmdline']):
    if p.pid==os.getpid():continue
    args=p.info.get('cmdline') or []
    assert str(source) not in args, 'Preserve the existing read client'
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
stdout=OUT/'actual-DT-worker-bindings-v49.long-wait.stdout.txt'
stderr=OUT/'actual-DT-worker-bindings-v49.long-wait.stderr.txt'
with stdout.open('xb') as out,stderr.open('xb') as err:
    process=subprocess.Popen([env['VENV_PYTHON'],'-u',str(source),'--native-env'],env=env,
        stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
report=dict(unix=time.time(),pid=process.pid,birth=psutil.Process(process.pid).create_time(),
    source=str(source),source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
    stdout=str(stdout),stderr=str(stderr),read_client_timeout_seconds=1800,
    model_calls=0,DT_calls=0,optimizer_calls=0,training_interruptions=0)
dest=OUT/'actual-DT-worker-bindings-v49-client.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
