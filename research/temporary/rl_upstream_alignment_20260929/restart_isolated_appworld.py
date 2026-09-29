"""Restart only the two stopped AppWorld pilots using official per-job roots."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit = root / 'receipts/upstream-alignment-20260929'
manifest = audit / 'two-gpu-pilots/manifest.json'
jobs = json.loads(manifest.read_text())
script = audit / 'run_two_gpu_training_pilot.sh'
subprocess.run(['bash','-n',str(script)],check=True)
(audit / 'two-gpu-pilots/manifest-before-output-isolation.json').write_text(manifest.read_text())
for index, old in enumerate(jobs):
    if old['task'] != 'AppWorld':
        continue
    assert not Path(f'/proc/{old["pid"]}').exists(), 'Prior job still exists'
    old.update(status='stopped_by_us_before_update', stop_reason='Two job roots shared official default_<worker_id> output directories; no checkpoint accepted')
    (Path(old['directory'])/'job.json').write_text(json.dumps(old,indent=2)+'\n')
    directory = audit / 'two-gpu-pilots-isolated' / f'{old["method"]}-AppWorld'
    directory.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ, DT_ROOT=old['release'], CUDA_VISIBLE_DEVICES=old['devices'],
        ENV_NAME='AppWorld', METHOD=old['method'], PILOT_OUTPUT_ROOT=str(directory.parent),
        APPWORLD_SERVER_OFFSET='0' if old['method']=='dt' else '17')
    with (directory/'train.log').open('wb') as log:
        proc = subprocess.Popen(['bash',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job = dict(old, pid=proc.pid, directory=str(directory), started_unix=time.time(),
               previous_attempt=str(Path(old['directory'])/'job.json'),
               status='started_bounded_isolated_root_pilot',
               launcher=str(script), launcher_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    job.pop('stop_reason',None)
    (directory/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    jobs[index] = job
manifest.write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps([job for job in jobs if job['task']=='AppWorld'],indent=2))
