"""Read the one owned diagnostic process, phases, original logs and resources."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
body = r'''
import hashlib,json,psutil,subprocess,time
from pathlib import Path
out=Path(OUT)
launch=json.loads((out/'launch.json').read_bytes())
result=dict(observed_unix=time.time(),launch=launch,processes=[],files={},ray_logs=[])
def get(p,limit=None):
 raw=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
             text=(raw if limit is None else raw[-limit:]).decode(errors='replace'))
try:
 driver=psutil.Process(launch['pid']); assert driver.create_time()==launch['birth']
 for p in [driver,*driver.children(recursive=True)]:
  try:
   result['processes'].append(dict(pid=p.pid,birth=p.create_time(),name=p.name(),
       status=p.status(),pss_bytes=p.memory_full_info().pss,cmdline=p.cmdline()))
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
except psutil.NoSuchProcess:pass
for name in ('phase.json','rank0.json','rank1.json','input-inspection.json','DT-input-inspection.json',
             'DT-output-observation.json','actor-input-with-native-old-ref.json'):
 p=out/name
 if p.exists():
  try:result['files'][name]=json.loads(p.read_bytes())
  except json.JSONDecodeError:result['files'][name]={'read_during_write':True}
for p in out.glob('rank[01]-*.json'):
 try:result['files'][p.name]=json.loads(p.read_bytes())
 except json.JSONDecodeError:pass
p=out/'driver.log'
if p.exists():result['driver_log']=get(p,18000)
own_pids={p['pid'] for p in result['processes']}
# Retain this diagnostic's original Ray logs after ray.shutdown as well.
phase=result['files'].get('phase.json',{})
if phase.get('unix',0)>=launch['launched_unix'] and phase.get('pid'):
 own_pids.add(phase['pid'])
for session in Path('/tmp/ray').glob('session_*'):
 if session.name.rsplit('_',1)[-1].isdigit() and int(session.name.rsplit('_',1)[-1]) in own_pids:
  for p in (session/'logs').glob('worker-*'):
   if p.suffix in ('.out','.err'):
    result['ray_logs'].append(get(p,22000))
result['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout
result['host_memory']=psutil.virtual_memory()._asdict()
result['cgroup']={p.name:p.read_text() for p in [Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.stat')] if p.exists()}
print(json.dumps(result))
'''
body = 'OUT='+repr(transport.ROOT+'/receipts/textcraft-DT-to-actor-nonfinite-20261008-v2')+'\n'+body
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=60)
(HERE/'observe-stderr.txt').write_bytes(result.stderr)
result.check_returncode()
value=json.loads(result.stdout)
folder=HERE/'observations';folder.mkdir(exist_ok=True)
receipt=folder/(str(int(value['observed_unix']))+'.json')
receipt.write_bytes(result.stdout)
for name,record in value['files'].items():
 if not record.get('read_during_write'):
  (folder/name).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
tail=value.get('driver_log',{}).get('text','')
dt_progress=[]
for log in value['ray_logs']:
 if log['path'].endswith('.out'):
  lines=[s for s in log['text'].splitlines() if s.startswith('[DT direct joint minibatch]')]
  if lines:dt_progress.append(dict(path=log['path'],latest=lines[-1]))
print(json.dumps(dict(receipt=str(receipt),observed_unix=value['observed_unix'],
    processes=len([p for p in value['processes'] if p['status']!='zombie']),
    phase=value['files'].get('phase.json'),
    ranks=[dict(rank=r,phase=v.get('phase'),microbatches=len(v.get('microbatches',[])),
                steps=len([s for s in v.get('optimizer_steps',[]) if s.get('completed_unix')]),
                error=v.get('error'),last_microbatch=v.get('microbatches',[])[-1:])
           for r in (0,1) if (v:=value['files'].get(f'rank{r}.json'))],
    tree_pss_bytes=sum(p['pss_bytes'] for p in value['processes']),
    DT_progress=dt_progress,
    log_tail='\n'.join(s for s in tail.splitlines()[-12:] if len(s)<1500)),ensure_ascii=False))
