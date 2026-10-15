"""Read existing capture, validation and deployment receipts without scheduling work."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
code=r'''
import json,time
from pathlib import Path
import psutil
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
d=r/'receipts/textcraft-efficiency-20261015-v1'
result=dict(unix=time.time(),receipts={},logs={},drivers={},files={})
names=['native-frozen-head-full-v2-client.json','native-frozen-head-full-v2-installation.json',
 'native-frozen-head-full-chunk.json','native-frozen-head-full-v2-error.json',
 'selected-head-verification-client.json','selected-head-verification-client-result.json',
 'selected-head-worker-verification.json','head-selection-verification.json',
 'DTPO-selected-head-deployment.json','PPO-selected-head-deployment.json']
for name in names:
 p=d/name
 if p.is_file():
  v=json.loads(p.read_bytes());result['receipts'][name]=v
  if 'pid' in v and 'birth' in v:
   try:v['process_matches']=psutil.Process(v['pid']).create_time()==v['birth']
   except psutil.NoSuchProcess:v['process_matches']=False
for name in ['native-frozen-head-full-v2-client.stderr.log','selected-head-verification.stderr.log',
 'DTPO-selected-head-deploy.stderr.log','PPO-selected-head-deploy.stderr.log']:
 p=d/name
 if p.is_file():result['logs'][str(p)]=p.read_text(errors='replace')[-6000:]
for name in ['native-frozen-head-full-chunk.pt']:
 p=d/name
 if p.is_file():result['files'][str(p)]=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime)
for label,name,pid,birth in [('DTPO','textcraft-dt-grpo-multiplier-variance-only-20261014-v3',3140624,1791993570.92),('PPO','textcraft-ppo-control-fixed-verl-fresh-20261014-v2',3946004,1792041120.19)]:
 launch=json.loads((r/'runs'/name/'launch.json').read_bytes())
 try:
  p=psutil.Process(pid);v=dict(pid=p.pid,birth=p.create_time(),expected_birth=birth,matches=p.create_time()==birth)
 except psutil.NoSuchProcess:v=dict(pid=pid,matches=False)
 result['drivers'][label]=v
 for session in Path('/tmp/ray').glob('session_*_'+str(pid)):
  for p in (session/'logs').glob('worker-*'):
   if p.suffix not in ('.out','.err'):continue
   data=p.read_text(errors='replace').splitlines()
   if not data:continue
   if any('TaskRunner' in line for line in data[:20]) or any('actor/grad_norm' in line for line in data[-100:]):
    result['logs'][str(p)]=[dict(line=i+1,text=data[i][-3500:]) for i in range(max(0,len(data)-5),len(data))]
print(json.dumps(result,ensure_ascii=False))
'''
script='set -e\nsource '+entry.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
p=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
result=json.loads(p.stdout)
(HERE/'selection-progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
