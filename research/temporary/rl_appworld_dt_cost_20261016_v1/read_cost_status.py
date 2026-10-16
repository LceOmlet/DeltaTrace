"""Read the isolated native cost probe and already emitted original stage data."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
code=r'''
import json,time,psutil
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
a=r/'receipts/appworld-dt-cost-breakdown-20261016-v1'
x=dict(unix=time.time(),files={},logs={},profiles={})
for name in ['launcher.json','job.json','observer-client.json','observation-installed.json','nested-observer-client.json','nested-observation-installed.json','same-input-observer-client.json','same-input-pair-installed.json']:
 p=a/name
 if p.is_file():
  v=json.loads(p.read_bytes());x['files'][name]=v
  if 'pid' in v:
   try:
    q=psutil.Process(v['pid']);v['alive']=q.create_time()==v['birth'] and q.status()!=psutil.STATUS_ZOMBIE
    if name=='job.json':v['workers']=[dict(pid=c.pid,birth=c.create_time(),phase=c.name()) for c in q.children(recursive=True) if 'WorkerDict' in c.name()]
   except psutil.NoSuchProcess:v['alive']=False
for p in list(a.glob('rank*-DT-cost*.json'))+list(a.glob('rank*-DT-partial.json'))+list(a.glob('rank*-nested-components*.json'))+list(a.glob('rank*-same-input-pair.json')):
 x['profiles'][p.name]=json.loads(p.read_bytes())
for p in [a/'launcher.stderr.log',a/'observer.stderr.log',a/'nested-observer.stderr.log',a/'same-input-observer.stderr.log',r/'runs/appworld-dt-cost-breakdown-20261016-v1/appworld/driver.log']:
 if p.is_file():x['logs'][str(p)]=p.read_text(errors='replace')[-2200:]
print(json.dumps(x))
'''
p=subprocess.run(e.SSH+['bash','-s'],input=('source '+e.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES= "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n').encode(),capture_output=True,timeout=120)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
x=json.loads(p.stdout);(HERE/'latest-status.json').write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(unix=x['unix'],files={k:{a:b for a,b in v.items() if a!='original_source_bindings'} for k,v in x['files'].items()},
 profiles={k:dict(status=v.get('status'),batches=len(v.get('batches',v.get('events',[]))),started_unix=v['started_unix'],seconds=v.get('original_trajectories_wall_seconds',v.get('wall_seconds'))) for k,v in x['profiles'].items()},logs=x['logs']),ensure_ascii=False))
