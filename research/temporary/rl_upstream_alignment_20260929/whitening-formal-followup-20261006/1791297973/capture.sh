/opt/conda/bin/python - <<'PY'

import hashlib,json,psutil,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');now=time.time()
out=root/'receipts/whitening-formal-followup-20261006'/str(int(now))
out.mkdir(parents=True,exist_ok=False)
result={'observed_unix':now,'scope':'Read-only driver birth, original logs and artifacts, actual worker stack, physical resources. No training, model, DT, HTTP, RPC, signal or production mutation.','tasks':{},'artifact_directory':str(out)}
for task,run_name,pid,birth in [('TextCraft','textcraft-official-whitening-20261006-v2',3269087,1791289021.47),('AppWorld','appworld-official-whitening-20261006-v2',3592468,1791291997.14)]:
 run=root/'runs'/run_name/(task.lower()+'-dt')
 job=json.loads((run/'job.json').read_bytes())
 row={'pid':pid,'expected_birth':birth,'run':str(run),'job':job,'pid_exists':psutil.pid_exists(pid),'logs':[],'stacks':[],'run_artifacts':[],'processes':[]}
 descendants=[]
 if row['pid_exists']:
  p=psutil.Process(pid);row['actual_birth']=p.create_time();row['birth_matches']=p.create_time()==birth
  assert row['birth_matches'],'PID reused; do not inspect as this job'
  row['driver_status']=p.status();row['driver_elapsed_seconds']=now-birth;descendants=p.children(recursive=True)
 for p in descendants:
  try:
   if 'TaskRunner' not in p.name() and 'WorkerDict' not in p.name():continue
   m=p.memory_full_info();row['processes'].append({'pid':p.pid,'birth':p.create_time(),'name':p.name(),'cpu_seconds':sum(p.cpu_times()[:2]),'pss_bytes':m.pss,'rss_bytes':m.rss,'status':p.status()})
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 sessions=list(Path('/tmp/ray').glob('session_*_'+str(pid)))
 session=max(sessions,key=lambda p:p.stat().st_mtime) if sessions else None
 row['ray_session']=str(session) if session else None
 logs=[run/'train.log']
 if session:
  for proc in row['processes']:
   logs.extend((session/'logs').glob('worker-*-'+str(proc['pid'])+'.out'))
   logs.extend((session/'logs').glob('worker-*-'+str(proc['pid'])+'.err'))
 for i,p in enumerate(dict.fromkeys(logs)):
  if not p.is_file():continue
  size=p.stat().st_size;start=max(0,size-160000)
  with p.open('rb') as f:f.seek(start);data=f.read(160000)
  dest=out/(task.lower()+'-raw-'+str(i)+p.suffix);dest.write_bytes(data)
  row['logs'].append({'source':str(p),'source_size_bytes':size,'mtime':p.stat().st_mtime,'snapshot':str(dest),'byte_range':[start,start+len(data)],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'tail_text':data.decode(errors='replace')[-1800:]})
 for p in sorted(run.iterdir()):
  if p.is_file():row['run_artifacts'].append({'name':p.name,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime})
 checkpoint=run/'checkpoints/latest_checkpointed_iteration.txt'
 row['checkpoint_marker']={'path':str(checkpoint),'exists':checkpoint.is_file(),'text':checkpoint.read_text() if checkpoint.is_file() else None}
 taskr=[p for p in row['processes'] if 'TaskRunner' in p['name']][:1]
 actors=[p for p in row['processes'] if 'WorkerDict' in p['name']][:2]
 for proc in taskr+actors:
  cmd=['/opt/conda/bin/py-spy','dump','--nonblocking','--json','--pid',str(proc['pid'])]
  try:
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=8)
   stack={'pid':proc['pid'],'birth':proc['birth'],'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'command':cmd}
  except subprocess.TimeoutExpired:stack={'pid':proc['pid'],'birth':proc['birth'],'observation_error':'py-spy timeout','command':cmd}
  row['stacks'].append(stack)
 result['tasks'][task]=row
result['cgroup']={}
for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.stat'):
 p=Path('/sys/fs/cgroup/memory')/name
 if p.is_file():result['cgroup'][name]=p.read_text()
result['host_available_bytes']=psutil.virtual_memory().available
try:
 r=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=10)
 result['physical_mx_smi']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired:result['physical_mx_smi']={'observation_error':'mx-smi timeout'}
result['collection_wall_seconds']=time.time()-now
p=out/'phase-observation.json';p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'artifact_directory':str(out),'record':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'observed_unix':now,'collection_wall_seconds':result['collection_wall_seconds'],'birth_matches':{k:v.get('birth_matches') for k,v in result['tasks'].items()}}),flush=True)

PY
