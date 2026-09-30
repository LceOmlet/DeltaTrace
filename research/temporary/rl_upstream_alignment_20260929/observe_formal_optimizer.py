"""Read original live optimizer state at an existing worker RPC boundary."""
from stage_environment_entry import remote,ROOT,ENTRY

remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil,time,ray
root=Path('@ROOT@')
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='TextCraft')
driver=psutil.Process(job['pid'])
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))

def observe(worker):
 import os,time,torch
 records=[]
 for role,w in worker.worker_dict.items():
  if not hasattr(w,'actor'):continue
  actor=w.actor;steps=[];finite=True;nonzero=elements=0
  for state in actor.actor_optimizer.state.values():
   step=state.get('step')
   if step is not None:steps.append(float(step.item() if hasattr(step,'item') else step))
   for name in ('exp_avg','exp_avg_sq'):
    value=state.get(name)
    if value is None:continue
    if hasattr(value,'to_local'):value=value.to_local()
    value=value.detach().cpu()
    finite=finite and bool(torch.isfinite(value).all())
    elements+=value.numel();nonzero+=int(value.count_nonzero())
  records.append(dict(pid=os.getpid(),rank=w.rank,unix=time.time(),role=role,
   optimizer_type=type(actor.actor_optimizer).__name__,optimizer_steps=sorted(set(steps)),
   state_entries=len(actor.actor_optimizer.state),moment_elements=elements,
   nonzero_moment_elements=nonzero,moments_finite=finite,
   microbatch=actor.config.ppo_micro_batch_size_per_gpu,
   rank_alpha=[dict(rank=c.r,alpha=c.lora_alpha) for c in actor.actor_module.peft_config.values()]))
 return records

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 names=[n for n in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in n['name']]
 assert len(names)==2,names
 refs=[ray.get_actor(n['name'],namespace=n['namespace']).execute_with_func_generator.remote(observe) for n in names]
 print(json.dumps(dict(task=job['task'],driver_pid=driver.pid,submitted_unix=time.time(),status='queued read-only RPC')),flush=True)
 result=dict(task=job['task'],driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  scope='Original optimizer state observation only; no weight mutation, numerical threshold, checkpoint or new update',
  workers=ray.get(refs))
 out=root/'receipts/owner-b8-dispatch-20260930/textcraft-formal-optimizer.json'
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
finally:ray.shutdown()
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
