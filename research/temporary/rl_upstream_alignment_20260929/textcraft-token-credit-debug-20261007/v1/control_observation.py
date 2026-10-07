"""Install a passive observer via the pinned VERL Worker RPC, then resume scheduling.

The existing eighth batch is reused. No training code, parameter, model replay,
sampling, or checkpoint operation is introduced. `resume-scheduler` never
resumes actor workers held before update; they require a later explicit action.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP

REMOTE = ROOT + '/receipts/textcraft-token-credit-debug-20261007-v1'
OBSERVER = HERE / 'observe_token_credit.py'
EXPECTED = dict(pid=110053, birth=1791344324.6, task_runner_pid=113691,
                task_runner_birth=1791344344.22,
                source_sha256='5013ebc878b972a5f52817f7c7de7ae55ddae7f1d61609e943e1ab55bf59b993',
                readout_sha256='31e2acfb760eb1dd118af78a5a887898357feacc1b0f1caec04f06006a3b7d07')

CODE = r'''
from pathlib import Path
import hashlib,json,os,psutil,sys,time
root=Path(@ROOT@);out=Path(@OUT@);expected=json.loads(@EXPECTED@)
job_entry=root/'candidates/direct-action-target-20261007-v3/textcraft/entry'
verl=root/'candidates/direct-action-target-20261007-v3/textcraft/verl'
source=root/'runs/direct-action-target-20261007-v3/textcraft/textcraft-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected['source_sha256']
driver=psutil.Process(expected['pid']);task_runner=psutil.Process(expected['task_runner_pid'])
assert driver.create_time()==expected['birth']
assert task_runner.create_time()==expected['task_runner_birth']
action=@ACTION@
record=dict(action=action,observed_unix=time.time(),expected=expected,
            source_path=str(source),observer_path=str(out/'observe_token_credit.py'),
            observer_sha256=hashlib.sha256((out/'observe_token_credit.py').read_bytes()).hexdigest())
if action=='install':
 assert driver.status()==psutil.STATUS_STOPPED and task_runner.status()==psutil.STATUS_STOPPED
 for path in reversed([str(job_entry),str(verl)]):sys.path.insert(0,path)
 os.environ['PYTHONPATH']=':'.join([str(job_entry),str(verl),os.environ.get('PYTHONPATH','')])
 import ray
 gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
 port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
 def install(worker):
  import hashlib,importlib.util,inspect,os,psutil,sys,time
  from pathlib import Path
  path=Path(@OUT@)/'observe_token_credit.py'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==@OBSERVER_SHA@
  module_name='textcraft_passive_token_credit_20261007_v1'
  spec=importlib.util.spec_from_file_location(module_name,path)
  module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
  owners=[];seen=set()
  for owner in worker.worker_dict.values():
   producer=getattr(owner,'_deltatrace_producer',None)
   if producer is None or getattr(producer,'direct_readout',None) is None or id(owner) in seen:continue
   seen.add(id(owner));readout=producer.direct_readout
   owner_path=Path(inspect.getsourcefile(type(readout)))
   owner_sha=hashlib.sha256(owner_path.read_bytes()).hexdigest()
   assert owner_sha==@READOUT_SHA@,(str(owner_path),owner_sha)
   assert owner.actor.config.ppo_micro_batch_size_per_gpu==4
   lora=[dict(rank=p.r,alpha=p.lora_alpha) for p in owner.actor.actor_module.peft_config.values()]
   assert all(p==dict(rank=8,alpha=16) for p in lora)
   owners.append(dict(rank=owner.rank,readout_source=str(owner_path),readout_sha256=owner_sha,
    original_update_source=inspect.getsourcefile(owner.update_actor),
    original_microbatch=owner.actor.config.ppo_micro_batch_size_per_gpu,lora=lora))
  assert len(owners)==1,owners
  provenance=dict(driver_pid=110053,driver_birth=1791344324.6,
   source_sha256=@SOURCE_SHA@,observer_sha256=@OBSERVER_SHA@,
   observer_path=str(path),owner=owners[0],purpose='Actual eighth batch credit debug before any actor update')
  rows=module.install_on_worker_dict(worker,@OUT@,provenance=provenance,hold=True)
  assert len(rows)==1
  return dict(pid=os.getpid(),birth=psutil.Process().create_time(),installed_unix=time.time(),
   owner=owners[0],rows=rows)
 ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
 try:
  actors=[n for n in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in n['name']]
  assert len(actors)==2,actors
  record['actors']=actors;record['gcs_port']=port
  (out/'install-submitted.json').write_text(json.dumps(record,indent=2)+'\n')
  refs=[ray.get_actor(n['name'],namespace=n['namespace']).execute_with_func_generator.remote(func=install) for n in actors]
  record['workers']=ray.get(refs,timeout=60)
  record['installed_unix']=time.time()
  (out/'installed.json').write_text(json.dumps(record,indent=2)+'\n')
 finally:ray.shutdown()
elif action=='resume-scheduler':
 installed=json.loads((out/'installed.json').read_text())
 assert installed['expected']==expected and installed['observer_sha256']==record['observer_sha256']
 assert sorted(w['owner']['rank'] for w in installed['workers'])==[0,1]
 for w in installed['workers']:
  process=psutil.Process(w['pid']);assert process.create_time()==w['birth']
  assert process.status()!=psutil.STATUS_STOPPED
 assert driver.status()==psutil.STATUS_STOPPED and task_runner.status()==psutil.STATUS_STOPPED
 record['resumed_scheduler']=[]
 for process in (task_runner,driver):
  process.resume();record['resumed_scheduler'].append(dict(pid=process.pid,birth=process.create_time()))
 record['policy']='Only scheduler resumed. Workers will self-hold BEFORE original update_actor.'
 (out/'resume-scheduler.json').write_text(json.dumps(record,indent=2)+'\n')
elif action=='check':
 installed=json.loads((out/'installed.json').read_text())
 record['processes']=[]
 for identity in [dict(pid=driver.pid,birth=driver.create_time()),dict(pid=task_runner.pid,birth=task_runner.create_time())]+[dict(pid=w['pid'],birth=w['birth']) for w in installed['workers']]:
  process=psutil.Process(identity['pid']);assert process.create_time()==identity['birth']
  record['processes'].append(dict(identity,status=process.status(),name=process.name()))
 record['artifacts']=[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in out.glob('rank*-*.pt')]
 for w in installed['workers']:
  pid=w['pid'];paths=list(Path('/tmp').glob(f'ray/**/worker-*-{pid}.out'))
  if not paths:paths=list(Path('/tmp').glob(f'**/logs/worker-*-{pid}.out'))
  record.setdefault('worker_tails',[]).append(dict(pid=pid,logs=[dict(path=str(p),tail=p.read_text(errors='replace')[-1800:]) for p in paths[-2:]]))
 memory=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
 record['cgroup_memory_bytes']=int(memory.read_text()) if memory.exists() else None
 stamp=int(time.time());(out/f'check-{stamp}.json').write_text(json.dumps(record,indent=2)+'\n')
else:raise ValueError(action)
print(json.dumps(record),flush=True)
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['install','resume-scheduler','check'])
    args=parser.parse_args()
    if args.action=='install':
        subprocess.run(SSH+['mkdir','-p',REMOTE],check=True)
        subprocess.run(SCP+[str(OBSERVER),'root@ssh.v5000-prod-gw.nhss.zhejianglab.com:'+REMOTE+'/observe_token_credit.py'],check=True)
    values={'ROOT':ROOT,'OUT':REMOTE,'EXPECTED':json.dumps(EXPECTED),'ACTION':args.action,
            'OBSERVER_SHA':hashlib.sha256(OBSERVER.read_bytes()).hexdigest(),
            'READOUT_SHA':EXPECTED['readout_sha256'],'SOURCE_SHA':EXPECTED['source_sha256']}
    code=CODE
    for key,value in values.items():code=code.replace('@'+key+'@',repr(value))
    shell='set -e\nsource '+ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+"\nPY\n"
    result=subprocess.run(SSH+['bash','-s'],input=shell.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (HERE/(args.action+'.stdout.txt')).write_bytes(result.stdout)
    (HERE/(args.action+'.stderr.txt')).write_bytes(result.stderr)
    if result.returncode:
        print(result.stderr.decode(errors='replace')[-6000:]);result.check_returncode()
    lines=result.stdout.decode().splitlines()
    record=json.loads(next(line for line in reversed(lines) if line.startswith('{')))
    (HERE/(args.action+'-receipt.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
