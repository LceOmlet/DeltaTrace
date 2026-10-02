"""Activate the minimal owner resource patch at the original worker RPC seam.

No worker pause/restart, model call, optimizer update, or training-config change.
The frozen launch files stay intact.  Each completion identifies the actual
patched owner methods, worker PID, and original driver creation time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SSH, SCP


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', required=True, choices=['AppWorld', 'TextCraft'])
args = parser.parse_args()
source = AUDIT / 'native-host-cache-owner-patched.py'
assert source.is_file()
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
metadata = json.loads((AUDIT / 'native-host-cache-source-before.json').read_text())
candidate = ROOT + '/candidates/native-host-cache-phase-20261002/' + source_sha[:12]

prepare = f'mkdir -p {candidate}\n'
subprocess.run(SSH + ['bash', '-s'], input=prepare.encode(), check=True, timeout=25)
subprocess.run(SCP + [str(source), f'{SSH[-1]}:{candidate}/fsdp_workers.py'],
               check=True, timeout=25)
script = r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,os,psutil,sys,time,ray
root=Path('@ROOT@');candidate=Path('@CANDIDATE@')/'fsdp_workers.py'
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='@TASK@')
driver=psutil.Process(job['pid'])
assert driver.create_time()==job['observed_process_created_unix']
for path in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,path)
os.environ['PYTHONPATH']=':'.join([job['entry'],job['verl_root'],os.environ.get('PYTHONPATH','')])
directory=root/'receipts/native-host-cache-phase-20261002'/f"{job['task']}-{job['pid']}"
directory.mkdir(parents=True,exist_ok=True)
assert not (directory/'submitted.json').exists(),'Inspect existing receipts instead of submitting twice'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='@SOURCE_SHA@'
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
expected_pids={p.pid for p in driver.children(recursive=True) if 'WorkerDict' in p.name()}
assert len(expected_pids)==2,expected_pids

def activate(worker):
 import ast,hashlib,inspect,json,os,time,torch
 from pathlib import Path
 from types import MethodType
 from verl.single_controller.base.decorator import MAGIC_ATTR
 assert os.getpid() in expected_pids
 assert hasattr(torch._C,'_host_emptyCache')
 source=candidate.read_text()
 assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='@SOURCE_SHA@'
 owner=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker')
 nodes={n.name:n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name in ('update_actor','compute_dt_token_advantages')}
 records=[]
 for role,w in worker.worker_dict.items():
  if not hasattr(w,'actor'):continue
  assert w.config.model.lora_rank==8 and w.config.model.lora_alpha==16
  assert w.actor.config.ppo_micro_batch_size_per_gpu==4
  before=(id(w.actor.actor_module),id(w.actor.actor_optimizer),repr(w.config))
  bindings={}
  for name,node in nodes.items():
   old=getattr(w,name).__func__;original=inspect.unwrap(old)
   prior=Path(inspect.getsourcefile(original))
   assert hashlib.sha256(prior.read_bytes()).hexdigest()=='@BEFORE_SHA@',str(prior)
   namespace=dict(original.__globals__)
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(candidate),'exec'),namespace)
   new=namespace[name]
   assert getattr(new,MAGIC_ATTR)==getattr(old,MAGIC_ATTR)
   new.__qualname__=old.__qualname__
   inspect.unwrap(new).__qualname__=original.__qualname__
   bindings[name]=dict(prior_source=str(prior),prior_sha256='@BEFORE_SHA@',effective_source=str(candidate),effective_sha256='@SOURCE_SHA@')
   setattr(w,name,MethodType(new,w))
  os.environ['VERL_RELEASE_UNUSED_HOST_CACHE']='1'
  assert before==(id(w.actor.actor_module),id(w.actor.actor_optimizer),repr(w.config))
  record=dict(task=job['task'],pid=os.getpid(),rank=w.rank,role=role,unix=time.time(),
   driver_pid=job['pid'],driver_created_unix=job['observed_process_created_unix'],methods=bindings,
   effective_forward_source=str(candidate),effective_source_sha256='@SOURCE_SHA@',
   resource_environment={'VERL_RELEASE_UNUSED_HOST_CACHE':'1'},
   submission_repository_commit='@REVISION@',submission_script_sha256='@SCRIPT_SHA@',
   lora_rank=8,lora_alpha=16,actor_microbatch=4,
   scope='Native unused-host-cache release only after whole DT/actor RPC cleanup; original loss, credit, microbatch loops, offload policy, model, optimizer and task config unchanged')
  (directory/f"rank{w.rank}.json").write_text(json.dumps(record,indent=2)+'\n')
  records.append(record)
 assert len(records)==1,records
 return records[0]

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[n for n in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in n['name']]
 assert len(actors)==2,actors
 submission=dict(task=job['task'],driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  worker_pids=sorted(expected_pids),submitted_unix=time.time(),source=str(candidate),source_sha256='@SOURCE_SHA@',
  submission_repository_commit='@REVISION@',submission_script_sha256='@SCRIPT_SHA@',
  status='Original sequential RPC submitted; per-rank receipts determine completion')
 (directory/'submitted.json').write_text(json.dumps(submission,indent=2)+'\n')
 refs=[ray.get_actor(n['name'],namespace=n['namespace']).execute_with_func_generator.remote(func=activate) for n in actors]
 ready,pending=ray.wait(refs,num_returns=2,timeout=15)
 records=ray.get(ready)
 print(json.dumps(dict(directory=str(directory),ready=len(ready),pending=len(pending),workers=records),indent=2),flush=True)
 if not pending:
  completed=dict(submission,completed_unix=time.time(),workers=records,status='Activated at original RPC seam')
  (directory/'complete.json').write_text(json.dumps(completed,indent=2)+'\n')
finally:ray.shutdown()
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',ROOT).replace('@TASK@',args.task)
script=script.replace('@CANDIDATE@',candidate).replace('@SOURCE_SHA@',source_sha)
script=script.replace('@BEFORE_SHA@',metadata['owner_sha256']).replace('@REVISION@',revision).replace('@SCRIPT_SHA@',script_sha)
result=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=55)
receipt=AUDIT/f"native-host-cache-activation-{args.task.lower()}-{int(time.time())}.txt"
receipt.write_bytes(result.stdout+result.stderr)
print(result.stdout.decode(errors='replace'))
print(result.stderr.decode(errors='replace'))
print('Receipt:',receipt)
raise SystemExit(result.returncode)
