"""Start bounded native two-iteration checks after owner inference assertions.

No trainer or environment algorithm here: publish a source overlay and call the
existing shell launcher. Two interaction steps expose generation after updates
without spending a full formal rollout just to test that interface.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit=root/'receipts/upstream-alignment-20260929'
release=root/'releases/7431001'
assert json.loads((audit/'vllm-hybrid-owner.json').read_text())['status']=='passed'
assert not release.exists(), 'Inspect existing release rather than overwrite it'
shutil.copytree(root/'releases/fc2e6c2',release,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(audit/'owner-verified-release.tar') as archive:
    archive.extractall(release,filter='data')
receipt=dict(revision='7431001',base_runtime=str(root/'releases/fc2e6c2'),
    purpose='Bounded owner-interface recheck, not formal budget or task performance',
    created_unix=time.time(),runtime_sha256={str(p.relative_to(release)):hashlib.sha256(p.read_bytes()).hexdigest()
        for sub in ('clean','accelerated','profiles','experiments/rl')
        for p in (release/sub).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json')})
(release/'deployment-owner-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
script=release/'research/temporary/rl_upstream_alignment_20260929/run_two_gpu_training_pilot.sh'
subprocess.run(['bash','-n',str(script)],check=True)
output=audit/'owner-verified-pilots'
output.mkdir(exist_ok=False)
jobs=[]
for method,task,devices,offset in [('dt','Sokoban','0,1',None),('dt','Webshop','2,3',None),
                                   ('dt','AppWorld','4,5',0),('grpo','AppWorld','6,7',17)]:
    directory=output/f'{method}-{task}'
    directory.mkdir()
    env=dict(os.environ,DT_ROOT=str(release),CUDA_VISIBLE_DEVICES=devices,ENV_NAME=task,METHOD=method,
        MAX_STEPS='2',PILOT_OUTPUT_ROOT=str(output),PYTHONPATH=str(audit),DT_VLLM_OBSERVE_DIR=str(directory))
    if offset is not None:
        env['APPWORLD_SERVER_OFFSET']=str(offset)
    args=['bash',str(script),'+ray_init.runtime_env.worker_process_setup_hook=observe_vllm_boundary.install',
          f'+ray_init.runtime_env.env_vars.DT_VLLM_OBSERVE_DIR={directory}']
    with (directory/'train.log').open('wb') as log:
        proc=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job=dict(method=method,task=task,devices=devices,pid=proc.pid,directory=str(directory),
        release=str(release),started_unix=time.time(),status='bounded_check_started_not_accepted',
        budget=dict(iterations=2,prompt_groups=4,group_size=4,max_steps=2,max_response=512,
                    context_cap=32768,optimizer_minibatch=64,actor_microbatch_per_gpu=4,dt_batch_per_gpu=4),
        command=args)
    (directory/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    jobs.append(job)
(output/'manifest.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps(jobs,indent=2))
