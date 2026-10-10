source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
TARGET='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/paper-role-author-curves-20261010-v1'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
INCLUDE_PHASES=False

import hashlib,json,os,psutil,subprocess,time
from pathlib import Path
root=Path(TARGET)
data={'unix':time.time(),'files':{},'processes':[]}
for name in ('launch.json','result.json','batching-cpu-replay.json','results/actor-initialization.json','results/completed.json',
             'results/rank0.json','results/rank1.json','results/rank0-first-nonfinite.json','results/rank1-first-nonfinite.json',
             'results/rank0-precast-seed.json','results/rank1-precast-seed.json'):
 p=root/name
 if p.exists():
  raw=p.read_bytes()
  try:value=json.loads(raw)
  except json.JSONDecodeError:value={'read_during_write':True,'bytes':len(raw)}
  if globals().get('SUMMARY_ONLY',False) and name.startswith('results/rank') and 'batches' in value:
   value={**{k:value.get(k) for k in ('phase','unix','elapsed_seconds','rank','pid','birth','operations','traceback')},
          'completed_points':sum(len(b.get('points',[])) for b in value['batches']),
          'batch_indices':[b['index'] for b in value['batches']], 'summary_only':True}
  data['files'][name]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'value':value}
launch=data['files'].get('launch.json',{}).get('value',{})
if launch:
 try:
  driver=psutil.Process(launch['pid'])
  assert driver.create_time()==launch['birth'],'PID was reused'
  for p in ([] if globals().get('PHASE_ONLY',False) else [driver]+driver.children(recursive=True)):
   try:
    info=p.as_dict(attrs=['pid','name','status','create_time','cpu_times'])
    info['cpu_times']=list(info['cpu_times']);m=p.memory_full_info()
    info.update(rss=m.rss,pss=getattr(m,'pss',None),uss=getattr(m,'uss',None))
    data['processes'].append(info)
   except psutil.NoSuchProcess:pass
  data['driver_alive']=True
 except psutil.NoSuchProcess:data['driver_alive']=False
for rank in (0,1):
 p=root/f'results/rank{rank}-phases.jsonl'
 if p.exists():
  lines=p.read_bytes().splitlines();events=[]
  for line in lines[-8:]:
   try:events.append(json.loads(line))
   except json.JSONDecodeError:pass
  data[f'rank{rank}_last_phases']=events
  if INCLUDE_PHASES:
   data[f'rank{rank}_all_phases']=[json.loads(line) for line in lines if line]
p=root/'driver.log'
if p.exists():data['driver_log_tail']='\n'.join(p.read_text(errors='replace').splitlines()[-80:])
data['physical']=('Not sampled: phase-only read' if globals().get('PHASE_ONLY',False)
                  else subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout)
data['host_memory']=psutil.virtual_memory()._asdict()
hold=Path(ROOT)/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
data['textcraft_hold']={'driver_pid':2833207,'expected_birth':1791370325.16,
 'release_exists':{str(r):(hold/f'rank{r}-release-update').exists() for r in (0,1)}}
try:data['textcraft_hold']['actual_birth']=psutil.Process(2833207).create_time()
except psutil.NoSuchProcess:data['textcraft_hold']['actual_birth']=None
data['cgroup']={}
for name in ('memory.usage_in_bytes','memory.stat'):
 p=Path('/sys/fs/cgroup/memory')/name
 if p.exists():data['cgroup'][name]=p.read_text()
print(json.dumps(data))

PY
