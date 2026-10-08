"""Read only this diagnostic's native logs, process tree and physical resources."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code=r'''
import hashlib,json,psutil,subprocess,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
value=dict(observed_unix=time.time(),launch=launch,files={},processes=[],ray_logs=[])
try:
 driver=psutil.Process(launch['pid']);assert driver.create_time()==launch['birth']
 for p in [driver,*driver.children(recursive=True)]:
  try:value['processes'].append(dict(pid=p.pid,birth=p.create_time(),status=p.status(),pss_bytes=p.memory_full_info().pss,cmdline=p.cmdline()))
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
except psutil.NoSuchProcess:pass
for path in out.glob('*.json'):
 if path.name=='launch.json':continue
 try:value['files'][path.name]=json.loads(path.read_bytes())
 except json.JSONDecodeError:value['files'][path.name]={'read_during_write':True}
def log(path):
 raw=path.read_bytes();return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),tail=raw[-20000:].decode(errors='replace'))
if (out/'driver.log').exists():value['driver_log']=log(out/'driver.log')
own={p['pid'] for p in value['processes']}
if value['files'].get('driver-state.json',{}).get('pid'):own.add(value['files']['driver-state.json']['pid'])
for session in Path('/tmp/ray').glob('session_*'):
 if session.name.rsplit('_',1)[-1].isdigit() and int(session.name.rsplit('_',1)[-1]) in own:
  for path in (session/'logs').glob('worker-*'):
   if path.suffix in ('.out','.err'):value['ray_logs'].append(log(path))
value['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout
value['host_memory']=psutil.virtual_memory()._asdict()
value['cgroup']={p.name:p.read_text() for p in [Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.stat')] if p.exists()}
print(json.dumps(value))
'''
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\nOUT='+repr(transport.ROOT+'/receipts/textcraft-native-first-iteration-nonfinite-20261008-v4')+'\n'+code+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=60)
(HERE/'read.stderr.txt').write_bytes(result.stderr)
if result.returncode:print(result.stderr.decode(errors='replace'))
result.check_returncode()
value=json.loads(result.stdout)
folder=HERE/'observations';folder.mkdir(exist_ok=True)
path=folder/(str(int(value['observed_unix']))+'.json');path.write_bytes(result.stdout)
for name,v in value['files'].items():
 if not v.get('read_during_write'):(folder/name).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
progress=[]
for log in value['ray_logs']:
 lines=[l for l in log['tail'].splitlines() if any(k in l for k in ('Rounds ','[DT direct joint minibatch]','grad_norm is not finite','step:1','Traceback','Error'))]
 if lines:progress.append(dict(path=log['path'],lines=lines[-3:]))
print(json.dumps(dict(observed_unix=value['observed_unix'],receipt=str(path),
 live_processes=sum(p['status']!='zombie' for p in value['processes']),
 tree_pss_bytes=sum(p['pss_bytes'] for p in value['processes']),
 ranks=[{k:v.get(k) for k in ('rank','phase','microbatches','optimizer_steps','error')} for r in (0,1) if (v:=value['files'].get(f'rank{r}.json'))],
 complete=value['files'].get('complete.json'),failure=value['files'].get('driver-failed.json'),
 progress=progress,driver_tail=value.get('driver_log',{}).get('tail','')[-2500:])))
