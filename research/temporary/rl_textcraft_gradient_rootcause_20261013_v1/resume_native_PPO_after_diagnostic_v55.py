"""Restore the paused PPO with native VERL checkpoint loading and original settings."""
import copy
import hashlib
import json
import os
from pathlib import Path
import psutil
import re
import subprocess
import sys
import time
from omegaconf import OmegaConf

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
stop=json.loads((OUT/'PPO-temporary-stop-for-v47-v54.json').read_bytes())
assert stop['status']=='temporarily_stopped_native_full_restore_required'
RUN=Path(stop['job']['run']);original=json.loads((RUN/'launch.json').read_bytes())
prepared=OUT/'native-PPO-resume-after-v47-v55-prepared.json'
destination=OUT/'native-PPO-resume-after-v47-v55-job.json'

if '--prepare' in sys.argv:
    assert not prepared.exists() and not destination.exists()
    options=copy.deepcopy(original['options'])
    options.update({'trainer.resume_mode':'resume_path','trainer.resume_from_path':stop['checkpoint']})
    changes={k:dict(before=original['options'].get(k),after=v) for k,v in options.items() if original['options'].get(k)!=v}
    assert set(changes)=={'trainer.resume_mode','trainer.resume_from_path'}
    argv=list(original['argv'])
    for key in changes:
        matching=[i for i,a in enumerate(argv) if a.startswith(key+'=')]
        assert len(matching)<=1
        value=key+'='+options[key]
        if matching:argv[matching[0]]=value
        else:argv.append(value)
    config=OmegaConf.load(Path(original['effective_environment']['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in options.items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    assert config.trainer.resume_mode=='resume_path' and config.trainer.resume_from_path==stop['checkpoint']
    report=dict(unix=time.time(),status='prepared_native_resume_only_not_submitted',run=str(RUN),
        argv=argv,options=options,effective_environment=original['effective_environment'],
        configuration_changes=changes,source_bindings=original['source_bindings'],
        original_launch_SHA256=hashlib.sha256((RUN/'launch.json').read_bytes()).hexdigest(),
        checkpoint=stop['checkpoint'],completed_iteration=stop['completed_iteration'],
        semantics='Original trainer loads actor, critic, optimizer, scheduler, RNG and dataloader; no state reconstruction.',
        no_new_training_hyperparameters=True,source_code_changes=0)
    prepared.write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'native-PPO-resume-after-v47-v55-effective-config.yaml').write_text(OmegaConf.to_yaml(config,resolve=True))
    print(json.dumps(dict(receipt=str(prepared),report=report)))
    raise SystemExit(0)

assert not destination.exists(), 'Inspect the existing resume job instead of launching twice'
record=json.loads(prepared.read_bytes())
diag=OUT/'native-common-update-feedback-v47'
result=json.loads((diag/'result.json').read_bytes())
assert result['phase']=='failed' or result['phase'].startswith('complete_'), 'Preserve running diagnostic'
physical=subprocess.check_output(['mx-smi'],text=True,timeout=30)
assert not re.search(r'\|\s*[23]\s+\d+\s+',physical.split('| Process:')[-1]), 'GPU2/3 still occupied'
for pid,birth in [(1419849,1791977138.31),(3174448,1791793629.73)]:
    p=psutil.Process(pid);assert p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE
for path,sha in record['source_bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
assert Path(record['checkpoint']).is_dir()
assert hashlib.sha256((Path(record['checkpoint'])/'data.pt').read_bytes()).hexdigest()==stop['data_SHA256']
env=dict(os.environ);env.update(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']='2,3'
for key in ('RAY_ADDRESS','RAY_TMPDIR'):env.pop(key,None)
stdout=RUN/f'driver.resume{record["completed_iteration"]}-v55.stdout.log'
stderr=RUN/f'driver.resume{record["completed_iteration"]}-v55.stderr.log'
with stdout.open('xb') as out,stderr.open('xb') as err:
    p=subprocess.Popen(record['argv'],cwd=env['VERL_ROOT'],env=env,stdout=out,stderr=err,start_new_session=True)
job=copy.deepcopy(stop['job'])
job.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),
    status='submitted_native_PPO_resume_pending_actual_restore',resume_from_path=record['checkpoint'],
    resumed_completed_iteration=record['completed_iteration'],previous_job_identity=stop['job'],
    resume_launch_receipt=str(prepared),stdout=str(stdout),stderr=str(stderr),
    user_authorized_temporary_stop_and_restore=True,hourly_automation='PAUSED',rootcause_repair_verified=False)
destination.write_text(json.dumps(job,indent=2)+'\n')
prior=OUT/'native-PPO-control-v9-job.before-resume-v55.json'
assert not prior.exists();prior.write_bytes((OUT/'native-PPO-control-v9-job.json').read_bytes())
(OUT/'native-PPO-control-v9-job.json').write_text(json.dumps(job,indent=2)+'\n')
(RUN/'resume-v55-job.json').write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(dict(receipt=str(destination),report=job)))
