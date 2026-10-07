"""Bounded read of exact paused job and original Ray failure records."""
from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH
code=r'''
from pathlib import Path
import hashlib,json,psutil,time,subprocess
root=Path('@ROOT@');out=root/'receipts/textcraft-token-credit-debug-20261007-v1'
record=dict(observed_unix=time.time(),processes=[],logs=[])
for pid in (110053,113691,115709,117229):
 try:
  p=psutil.Process(pid);record['processes'].append(dict(pid=pid,birth=p.create_time(),status=p.status(),name=p.name(),children=[dict(pid=c.pid,name=c.name(),status=c.status()) for c in p.children()]))
 except psutil.NoSuchProcess:record['processes'].append(dict(pid=pid,exists=False))
log=root/'runs/direct-action-target-20261007-v3/textcraft/textcraft-dt/train.log'
paths=[log]
sessions=sorted(Path('/tmp').glob('ray/session_*'))
for session in sessions:
 logs=session/'logs'
 originals=list(logs.glob('*113691*'))+list(logs.glob('*115709*'))+list(logs.glob('*117229*'))
 if originals:
  paths+=originals
  paths += [logs/'raylet.out',logs/'raylet.err',logs/'gcs_server.out']
for p in dict.fromkeys(paths):
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-20000));tail=f.read().decode(errors='replace')
  record['logs'].append(dict(path=str(p),bytes=p.stat().st_size,tail=tail))
memory=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
record['cgroup_memory_bytes']=int(memory.read_text()) if memory.exists() else None
record['physical_gpu']=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout
path=out/f"paused-terminal-{int(time.time())}.json";path.write_text(json.dumps(record,indent=2)+'\n')
record['remote_receipt']=str(path);print(json.dumps(record),flush=True)
'''.replace('@ROOT@',ROOT)
shell='set -e\nsource '+ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+"\nPY\n"
r=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,check=True)
d=json.loads(r.stdout.decode().splitlines()[-1]);p=HERE/f"paused-terminal-{int(d['observed_unix'])}.json"
p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(receipt=str(p),processes=d['processes'],log_files=[v['path'] for v in d['logs']],physical_gpu=d['physical_gpu']),ensure_ascii=False))
