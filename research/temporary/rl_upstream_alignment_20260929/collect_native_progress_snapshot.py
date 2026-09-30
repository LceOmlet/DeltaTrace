"""Read native worker output and physical/cgroup resources of current jobs."""
from stage_environment_entry import remote,ROOT,ENTRY
remote(fr'''source {ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,time,psutil,re,subprocess
root=Path('{ROOT}');active=json.loads((root/'active-training.json').read_text());out=Path(active['manifest']).parent
records=[]
for j in active['jobs']:
 try:p=psutil.Process(j['pid'])
 except psutil.NoSuchProcess:
  record=dict(task=j['task'],pid=j['pid'],live=False,log=j['log'])
  records.append(record);print(json.dumps(record));continue
 files=set();pss=0
 for child in [p]+p.children(recursive=True):
  try:
   pss+=child.memory_full_info().pss
   files.update(f.path for f in child.open_files() if '/worker-' in f.path and f.path.endswith(('.out','.err')))
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 found=[];credit_groups=[]
 for name in sorted(files):
  s=Path(name).read_text(errors='replace')
  groups=[]
  for line in s.splitlines():
   if line.startswith('[DT EOS plan] '):
    groups.append(dict(plan=line,finished=False,latest_batch=None,observed_batch_seconds=0.))
   elif groups and line.startswith('[DT EOS minibatch] '):
    group=groups[-1];group['latest_batch']=line
    duration=re.search(r' seconds=([\d.]+)',line)
    if duration:group['observed_batch_seconds']+=float(duration.group(1))
   elif groups and line.startswith('[DeltaTrace readout] '):
    value=json.loads(line.split('] ',1)[1]);groups[-1]['finished']=True
    groups[-1]['report']={{k:v for k,v in value.items() if k not in ('traces','minimum_log_ratio_batch','actual_row_lengths')}}
  if groups:credit_groups.append(dict(path=name,groups=groups,
   scope='Observed worker RPC groups only; later reward-alphabet groups may not yet have been submitted. A batch=N/M counter describes this group, not the whole iteration.'))
  matches=[l for l in s.splitlines() if re.search(r'^\[loop_transport\]|^\[loop_trajectory\]|^\[owner_trajectory\]|^\[DT EOS |^\[DT rollout\]|^\[DeltaTrace readout\]|^Rounds \d|n_rollouts_collected=|Cancelling rollout|Rollout collection completed|^step:|^Traceback|OutOfMemoryError|FloatingPointError|AssertionError',l)]
  brief=[]
  for line in matches[-6:]:
   if line.startswith('[DT EOS minimum] '):
    value=json.loads(line.split('] ',1)[1]);line='[DT EOS minimum traces] '+json.dumps([item['trace'] for item in value['samples']])
   elif line.startswith('[DeltaTrace readout] '):
    value=json.loads(line.split('] ',1)[1]);line='[DeltaTrace readout] '+json.dumps({{k:v for k,v in value.items() if k not in ('traces','minimum_log_ratio_batch','actual_row_lengths')}})
   brief.append(line)
  if brief:found.append(dict(path=name,lines=brief))
 record=dict(task=j['task'],method=j['method'],pid=p.pid,live=p.is_running(),age_s=round(time.time()-p.create_time()),
  pss_gib=round(pss/2**30,2),evidence=found,dt_credit_groups=credit_groups,checkpoints=[dict(path=str(f),value=f.read_text()) for f in Path(j['checkpoints']).glob('latest*')])
 records.append(record);print(json.dumps(record))
usage=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
smi=subprocess.check_output(['mx-smi'],text=True)
result=dict(unix=time.time(),jobs=records,cgroup_gib=usage/2**30,physical_gpu=smi)
(out/'progress-current.json').write_text(json.dumps(result,indent=2)+'\n')
print('cgroup_gib',round(usage/2**30,2))
for l in smi.splitlines():
 if 'MiB' in l and '/' in l:print(l)
PY
''')
