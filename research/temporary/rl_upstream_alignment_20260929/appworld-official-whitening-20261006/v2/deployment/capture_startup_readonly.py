"""One bounded read-only AppWorld startup snapshot; no training RPC or signal."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[2]
sys.path.insert(0,str(AUDIT))
from stage_environment_entry import ROOT,SCP,SSH


REMOTE_CODE=r'''
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path(@ROOT@);run=root/'runs/appworld-official-whitening-20261006-v2/appworld-dt'
now=time.time();out=root/'receipts/appworld-official-whitening-20261006-v2'/('startup-'+str(int(now)))
out.mkdir(parents=True,exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def metadata(p):
 p=Path(p)
 if not p.is_file():return dict(path=str(p),exists=False)
 data=p.read_bytes();s=p.stat()
 return dict(path=str(p),exists=True,bytes=len(data),sha256=sha(data),mtime=s.st_mtime)
job=json.loads((run/'job.json').read_bytes());source=json.loads((run/'source.json').read_bytes())
plan_path=root/'receipts/appworld-official-whitening-20261006-v2/launch-plan.json'
plan=json.loads(plan_path.read_bytes());pid=3592468;birth=1791291997.14
record=dict(observed_unix=now,scope='One bounded read-only actual startup. No model/test, GPU computation, service request, training RPC, signal, stop, retry or manifest mutation.',
 expected_pid=pid,expected_birth=birth,pid_exists=psutil.pid_exists(pid),artifact_directory=str(out),
 job=dict(source=metadata(run/'job.json'),content=job),source=dict(source=metadata(run/'source.json'),content=source),
 plan=dict(source=metadata(plan_path),content=plan),job_argv_equals_prepared_plan=job['argv']==plan['argv'])
if (run/'launch.json').is_file():
 launch=json.loads((run/'launch.json').read_bytes())
 prepared=json.loads((root/'receipts/appworld-official-whitening-20261006-v2/prepared.json').read_bytes())
 record['actual_driver_configuration']=dict(source=metadata(run/'launch.json'),options=launch['options'],
  equals_prepared_formal_options=launch['options']==prepared['unchanged_formal_options'])
else:record['actual_driver_configuration']=dict(source=metadata(run/'launch.json'),state='not yet emitted')
record['actual_frozen_sources']={}
names=[('trainer',Path(job['verl_root'])/'verl/trainer/ppo/ray_trainer.py'),
 ('helper',Path(job['verl_root'])/'verl/utils/torch_functional.py'),('actor',Path(job['verl_root'])/'verl/workers/actor/dp_actor.py'),
 ('worker',Path(job['verl_root'])/'verl/workers/fsdp_workers.py'),('core',Path(job['verl_root'])/'verl/trainer/ppo/core_algos.py'),
 ('readout',Path(job['entry'])/'reward_readout.py'),('producer',Path(job['entry'])/'deltatrace_rollout.py'),
 ('prefix_leases',Path(job['entry'])/'native_prefix_leases.py'),('loop_worker',Path(job['entry'])/'loop_owner_worker.py'),
 ('loop_rollout',Path(job['entry'])/'loop_owner_rollout.py'),('async_transport',Path(job['entry'])/'loop_async_transport.py'),
 ('loop_interface',Path(source['loop_root'])/'phi_agents/appworld/interface.py')]
for name,p in names:record['actual_frozen_sources'][name]=metadata(p)
descendants=[];driver=None
if record['pid_exists']:
 driver=psutil.Process(pid);record.update(actual_birth=driver.create_time(),birth_matches=driver.create_time()==birth,
  driver_status=driver.status(),driver_elapsed_seconds=now-driver.create_time())
 assert record['birth_matches'],'Do not inspect a reused PID as this task'
 environ=driver.environ()
 record['driver_configured_environment']={k:dict(expected=v,actual=environ.get(k),equal=environ.get(k)==v)
  for k,v in plan['environment'].items() if k in ('VERL_ROOT','DT_ROOT','DT_ENTRY_ROOT','LOOP_ROOT','APPWORLD_ROOT','CUDA_VISIBLE_DEVICES',
    'PYTHONPATH','VERL_RELEASE_UNUSED_HOST_CACHE','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
 descendants=driver.children(recursive=True)
record['processes']=[];selected=[]
for p in descendants:
 try:
  command=p.cmdline();name=p.name();is_loop=any('phi_agents.appworld.server' in x for x in command)
  row=dict(pid=p.pid,birth=p.create_time(),name=name,status=p.status(),cpu_seconds=sum(p.cpu_times()[:2]),rss_bytes=p.memory_info().rss,
   cwd=p.cwd(),native_loop_service=is_loop)
  if any(x in name for x in ('TaskRunner','WorkerDict')) or is_loop:
   row['pss_bytes']=p.memory_full_info().pss
   row['configured_environment']={k:p.environ().get(k) for k in ('DT_ENTRY_ROOT','VERL_ROOT','DT_ROOT','LOOP_ROOT','CUDA_VISIBLE_DEVICES','VERL_RELEASE_UNUSED_HOST_CACHE')}
  if is_loop:
   row['module']='phi_agents.appworld.server'
   row['stdout_fd_target']=os.readlink('/proc/'+str(p.pid)+'/fd/1')
  record['processes'].append(row)
 except (psutil.NoSuchProcess,psutil.AccessDenied,OSError) as exc:record['processes'].append(dict(pid=p.pid,observation_error=str(exc)))
ray=list(Path('/tmp/ray').glob('session_*_'+str(pid)))
session=sorted(ray,key=lambda p:p.stat().st_mtime)[-1] if ray else None
record['actual_ray_session']=str(session) if session else None
pattern=re.compile(r'Loading from |Setting global step|Resuming from |Training Progress|loop_|LOOP|Starting .*server|Uvicorn|Application startup|Running rounds|Rounds \d|n_rollouts_collected=|Traceback|Error executing|ReadTimeout|Could not connect|ready|actor/grad_norm|step:')
record['log_snapshots']=[]
logs=[run/'train.log']
if session:
 for p in descendants:
  if any(x in p.name() for x in ('TaskRunner','WorkerDict')):
   logs.extend((session/'logs').glob('worker-*-'+str(p.pid)+'.out'))
   logs.extend((session/'logs').glob('worker-*-'+str(p.pid)+'.err'))
for index,p in enumerate(dict.fromkeys(logs)):
 if not p.is_file():continue
 data=p.read_bytes();dest=out/('raw-'+str(index)+'-'+p.name);dest.write_bytes(data)
 lines=data.decode(errors='replace').splitlines()
 record['log_snapshots'].append(dict(path=str(p),bytes=len(data),snapshot_sha256=sha(data),snapshot_path=str(dest),
  line_count=len(lines),matching_lines=[dict(line=i,text=l) for i,l in enumerate(lines,1) if pattern.search(l)][-160:],
  original_byte_range=[0,len(data)],scope='Exact bytes read once; live original file can append after collection'))
record['worker_visibility']={}
visibility=run/'worker-visibility'
if visibility.is_dir():
 for p in visibility.glob('*.json'):
  data=p.read_bytes();dest=out/('visibility-'+p.name);dest.write_bytes(data)
  record['worker_visibility'][str(p)]=dict(sha256=sha(data),snapshot_path=str(dest),content=json.loads(data))
py_spy=Path('/opt/conda/bin/py-spy');record['py_spy']=metadata(py_spy);record['stack_dumps']=[]
task=[p for p in record['processes'] if 'TaskRunner' in p.get('name','')]
actors=[p for p in record['processes'] if 'WorkerDict' in p.get('name','')][:2]
loop=[p for p in record['processes'] if p.get('native_loop_service')][:1]
for row in task[:1]+actors+loop:
 if not py_spy.is_file():break
 command=[str(py_spy),'dump','--nonblocking','--json','--pid',str(row['pid'])]
 try:
  result=subprocess.run(command,capture_output=True,text=True,timeout=10)
  entry=dict(pid=row['pid'],actual_birth=row['birth'],command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
 except subprocess.TimeoutExpired as exc:entry=dict(pid=row['pid'],actual_birth=row['birth'],command=command,observation_error='py-spy timeout',stdout=exc.stdout.decode(errors='replace') if isinstance(exc.stdout,bytes) else exc.stdout,stderr=exc.stderr.decode(errors='replace') if isinstance(exc.stderr,bytes) else exc.stderr)
 record['stack_dumps'].append(entry)
 (out/('stack-'+str(row['pid'])+'.json')).write_text(json.dumps(entry,indent=2)+'\n')
record['native_loop_listening_ports']=[]
try:
 loop_pids={p['pid'] for p in record['processes'] if p.get('native_loop_service')}
 record['native_loop_listening_ports']=[dict(pid=c.pid,address=list(c.laddr),status=c.status) for c in psutil.net_connections(kind='inet') if c.pid in loop_pids and c.status=='LISTEN']
except (psutil.AccessDenied,OSError) as exc:record['port_observation_error']=str(exc)
try:
 result=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=10)
 record['physical_mx_smi']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
except subprocess.TimeoutExpired:record['physical_mx_smi']=dict(observation_error='mx-smi timeout')
record['host_available_bytes']=psutil.virtual_memory().available
cgroup=Path('/sys/fs/cgroup/memory');record['cgroup']={}
for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.stat'):
 p=cgroup/name
 if p.is_file():record['cgroup'][name]=p.read_text()
record['operations']=dict(model_initializations=0,DT_calls=0,rollout_calls=0,backward_calls=0,optimizer_steps=0,training_RPC_calls=0,service_HTTP_calls=0,signals=0)
record['collection_wall_seconds']=time.time()-now
target=out/'startup-phase-observation.json';target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(observed_unix=now,artifact_directory=str(out),record=str(target),record_sha256=sha(target.read_bytes()),
 birth_matches=record.get('birth_matches'),elapsed_seconds=record.get('driver_elapsed_seconds'),process_count=len(record['processes']),
 checkpoint_loading_lines=[l for f in record['log_snapshots'] for l in f['matching_lines'] if 'Loading from ' in l['text'] or 'Setting global step' in l['text']],
 collection_wall_seconds=record['collection_wall_seconds'])),flush=True)
'''


if __name__=='__main__':
    import ast
    code=REMOTE_CODE.replace('@ROOT@',repr(ROOT));ast.parse(code)
    script="/opt/conda/bin/python - <<'PY'\n"+code+"\nPY\n"
    (HERE/'startup-readonly.sh').write_text(script,encoding='utf-8')
    result=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    (HERE/'startup-readonly.stdout.txt').write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace'));print(result.stderr.decode(errors='replace'))
    result.check_returncode();data=json.loads(result.stdout.decode().splitlines()[-1])
    local=HERE/Path(data['artifact_directory']).name;local.mkdir(exist_ok=False)
    subprocess.run(SCP+['-r',f"{SSH[-1]}:{data['artifact_directory']}/.",str(local)],check=True)
