"""Read native PPO resume logs and checkpoint-state contents; no model calls."""
import hashlib
import json
from pathlib import Path
import psutil
import time
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
job_path=OUT/'native-PPO-resume-after-v47-v55-job.json'
job=json.loads(job_path.read_bytes())
alias=OUT/'native-PPO-control-v55-job.json'
if alias.is_symlink():assert alias.resolve()==job_path
else:assert not alias.exists();alias.symlink_to(job_path.name)
p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
report=dict(unix=time.time(),job=job,children=[dict(pid=c.pid,birth=c.create_time(),name=c.name())
    for c in p.children(recursive=True) if c.name().startswith('ray::WorkerDict')],logs=[],extra_states=[],
    model_calls=0,optimizer_calls=0)
client_path=OUT/'PPO-restore-binding-v55-client.json'
if client_path.exists():
    client=json.loads(client_path.read_bytes());binding=Path(client['result'])
    report['actual_worker_binding']=json.loads(binding.read_bytes()) if binding.exists() else None
    report['binding_reader_stderr']=Path(client['stderr']).read_text(errors='replace')[-1800:]
later=Path(job['run'])/'runtime-overrides/native-PPO-bindings-v56.json'
report['actual_native_rollout_RNG_owner_binding']=json.loads(later.read_bytes()) if later.exists() else None
paths=[Path(job['stdout']),Path(job['stderr'])]
for session in Path('/tmp/ray').glob('session_*_'+str(job['pid'])):
    paths.extend((session/'logs').glob('worker-*.out'))
    paths.extend((session/'logs').glob('worker-*.err'))
for path in paths:
    if not path.exists():continue
    for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        if any(s in line for s in ('Load from checkpoint','Setting global step','Resuming from ',
            'global_step_15/','No dataloader state found','Training Progress:','Traceback','OutOfMemoryError')):
            report['logs'].append(dict(path=str(path),line=number,text=line[-2200:]))
for role in ('actor','critic'):
    for rank in (0,1):
        path=Path(job['resume_from_path'])/role/f'extra_state_world_size_2_rank_{rank}.pt'
        value=torch.load(path,map_location='cpu',weights_only=False)
        report['extra_states'].append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            keys=list(value),scheduler=value.get('lr_scheduler'),rng_keys=list(value.get('rng',{})),
            rollout_rng_present=value.get('rollout_gen_random_states') is not None,
            rollout_rng_storage_key='rollout_gen_random_states'))
assert not torch.cuda.is_initialized()
dest=OUT/'PPO-resume-native-logs-v55.json';dest.write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report),default=str))
