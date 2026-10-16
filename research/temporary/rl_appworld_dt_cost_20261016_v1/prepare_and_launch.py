"""One original native AppWorld iteration with passive DT cost accounting only."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='appworld-dt-cost-breakdown-20261016-v1'
AUDIT=ROOT/'receipts'/VERSION
RUN=ROOT/'runs'/VERSION/'appworld'
assert not (AUDIT/'job.json').exists(),'Preserve the existing bounded trial'
base=json.loads((ROOT/'candidates/appworld-sql-native-speed-20261016-v3/AppWorld/prepared.json').read_bytes())
for path,sha in base['source_bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
options={k:v.replace(base['run'],str(RUN)) if isinstance(v,str) else v for k,v in base['options'].items()}
options['trainer.experiment_name']=VERSION
options['trainer.total_training_steps']=1
env={k:v.replace(base['run'],str(RUN)) if isinstance(v,str) else v for k,v in base['effective_environment'].items()}
env['CUDA_VISIBLE_DEVICES']='6,7';env.pop('MACA_VISIBLE_DEVICES',None)
cpu=dict(env,CUDA_VISIBLE_DEVICES='')
code='import sys,json;sys.path.insert(0,sys.argv[1]);from owner_runtime_options import owner_command;print(json.dumps(owner_command(json.loads(sys.stdin.read()))))'
argv=json.loads(subprocess.check_output([env['VENV_PYTHON'],'-c',code,base['entry']],input=json.dumps(options),env=cpu,text=True))
cfg=subprocess.run(argv+['--cfg','job','--resolve'],cwd=base['entry'],env=cpu,capture_output=True)
(AUDIT/'effective-config.yaml').write_bytes(cfg.stdout);(AUDIT/'CPU-config.stderr.log').write_bytes(cfg.stderr);cfg.check_returncode()
from omegaconf import OmegaConf
v=OmegaConf.create(cfg.stdout.decode())
assert v.trainer.total_training_steps==1 and v.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==4
assert v.actor_rollout_ref.rollout.max_model_len==32768 and v.algorithm.adv_estimator=='deltatrace'
assert v.trainer.save_freq==-1 and v.trainer.test_freq==-1
assert v.actor_rollout_ref.model.lora_rank==8 and v.actor_rollout_ref.model.lora_alpha==16
changes={k:dict(before=base['options'].get(k),after=value) for k,value in options.items() if value!=base['options'].get(k)}
assert set(changes)<= {'trainer.experiment_name','trainer.total_training_steps','trainer.default_local_dir',
 'trainer.rollout_data_dir','trainer.validation_data_dir','+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS'}
prepared=dict(base,version=VERSION,run=str(RUN),options=options,effective_environment=env,argv=argv,
              changes=changes,scope='One original native iteration; passive profiling, no method change')
(AUDIT/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
smi=subprocess.check_output(['mx-smi'],text=True)
(AUDIT/'physical-GPU-before-launch.txt').write_text(smi)
assert not any(re.match(r'\|\s*(?:6|7)\s+\d+\s+\S+',line) for line in smi.splitlines()),'GPU6/7 occupied; no process was stopped'
RUN.mkdir(parents=True,exist_ok=False)
(RUN/'launch.json').write_text(json.dumps(prepared,indent=2)+'\n')
with (RUN/'driver.log').open('xb') as stream:
    p=subprocess.Popen(argv,cwd=RUN,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),run=str(RUN),log=str(RUN/'driver.log'),
         started_unix=time.time(),physical_gpus=[6,7],formal=False,budget=1,
         original_source_bindings=base['source_bindings'],status='native_initialization')
job.update(profiling_source_commit=os.environ.get('DT_COST_SOURCE_COMMIT'),
    baseline_v5_commit=base['baseline_v5_commit'],upstream_commit='20bd331b',
    profiling_source_SHA256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__),AUDIT/'source/observe_native_DT.py')})
for path in (AUDIT/'job.json',RUN/'job.json'):path.write_text(json.dumps(job,indent=2)+'\n')
with (AUDIT/'observer.stdout.log').open('xb') as out,(AUDIT/'observer.stderr.log').open('xb') as err:
    client=subprocess.Popen([env['VENV_PYTHON'],'-u',str(AUDIT/'source/observe_native_DT.py')],env=env,
        cwd=AUDIT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
(AUDIT/'observer-client.json').write_text(json.dumps(dict(pid=client.pid,birth=psutil.Process(client.pid).create_time()),indent=2)+'\n')
print(json.dumps(job),flush=True)
exit_code=p.wait()
job.update(exit_code=exit_code,ended_unix=time.time(),status='native_trial_completed' if exit_code==0 else 'native_trial_failed')
for path in (AUDIT/'job.json',RUN/'job.json'):path.write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(dict(status=job['status'],exit_code=exit_code)),flush=True)
