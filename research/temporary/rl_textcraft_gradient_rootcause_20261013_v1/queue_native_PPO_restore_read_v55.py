"""Run the existing native PPO binding reader against the resumed identity."""
import hashlib
import json
import os
from pathlib import Path
import psutil
import subprocess
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
version=sys.argv[1] if len(sys.argv)>1 else 'v55'
assert version in ('v55','v56')
alias=OUT/f'native-PPO-control-{version}-job.json'
target=OUT/'native-PPO-resume-after-v47-v55-job.json'
if version=='v56' and not alias.exists():alias.symlink_to(target.name)
job=json.loads(alias.read_bytes())
RUN=Path(job['run']);launch=json.loads((RUN/'launch.json').read_bytes())
source=OUT/'verify_native_PPO_bindings_v8.py'
assert not (OUT/f'PPO-restore-binding-{version}-client.json').exists()
assert not (RUN/f'runtime-overrides/native-PPO-bindings-{version}.json').exists()
p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
workers=[c for c in p.children(recursive=True) if c.name().startswith('ray::WorkerDict')]
assert len(workers)==2 and all('init_model' not in w.name() for w in workers), 'Wait for native initialization'
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
stdout=OUT/f'PPO-restore-binding-{version}.stdout.txt';stderr=OUT/f'PPO-restore-binding-{version}.stderr.txt'
with stdout.open('xb') as out,stderr.open('xb') as err:
    client=subprocess.Popen([env['VENV_PYTHON'],'-u',str(source),version],env=env,
        stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
report=dict(unix=time.time(),pid=client.pid,birth=psutil.Process(client.pid).create_time(),
    source=str(source),source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
    stdout=str(stdout),stderr=str(stderr),result=str(RUN/f'runtime-overrides/native-PPO-bindings-{version}.json'),
    model_calls=0,optimizer_calls=0)
dest=OUT/f'PPO-restore-binding-{version}-client.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
