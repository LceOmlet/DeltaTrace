"""Read current job identity and original retry evidence before a bounded probe."""
import json
import subprocess
from stage_environment_entry import AUDIT, ENTRY, ROOT, SSH


script = r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,pathlib,psutil,re,subprocess,time
root=pathlib.Path('@ROOT@'); active=json.loads((root/'active-training.json').read_bytes())
report=dict(observed_unix=time.time(),manifest_sha256=hashlib.sha256((root/'active-training.json').read_bytes()).hexdigest(),jobs=[])
for job in active['jobs']:
 row=dict(task=job['task'],pid=job['pid'],expected_birth=job.get('observed_process_created_unix'),output=job['output'])
 try:
  p=psutil.Process(job['pid']); row.update(alive=p.is_running(),birth=p.create_time(),title=p.name())
  row['birth_matches']=p.create_time()==row['expected_birth']
  children=p.children(recursive=True) if row['birth_matches'] else []
 except psutil.NoSuchProcess:
  row['alive']=False;children=[]
 marker=pathlib.Path(job['checkpoints'])/'latest_checkpointed_iteration.txt'
 row['completed_checkpoint_marker']=marker.read_text().strip() if marker.exists() else None
 paths={job['log']}
 for child in children:
  try:
   if 'TaskRunner' in child.name():
    paths.update(f.path for f in child.open_files() if '/worker-' in f.path and f.path.endswith('.out'))
  except psutil.NoSuchProcess: pass
 row['logs']=[]
 for path in paths:
  p=pathlib.Path(path)
  if not p.exists():continue
  last=None;retry=[];terminal=[]
  with p.open(errors='replace') as stream:
   for number,line in enumerate(stream,1):
    if re.search(r'(?:^|\x1b\[[^m]*m | )step:\d+ -',line):last=line.strip()
    if 'Caught exception while running vllm inference' in line:retry.append(dict(line=number,text=line.strip()))
    if any(key in line for key in ('AssertionError','loop_owner_rollout.py','world.restart')):terminal.append(dict(line=number,text=line.strip()))
  row['logs'].append(dict(path=path,last_completed_metrics=last,retry_matches=retry[-20:],retry_match_count=len(retry),alignment_matches=terminal[-8:]))
 if job['task']=='AppWorld':
  source=pathlib.Path(job['output'])/'source.json';value=json.loads(source.read_bytes())
  row['source']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),loop_root=value.get('loop_root'))
  launch=pathlib.Path(job['output'])/'launch.json';value=json.loads(launch.read_bytes())
  row['launch_top_keys']=list(value)
  row['launch_environment']={k:v for k,v in value.get('environment',value.get('env',{})).items() if k in ('RAY_TMPDIR','LOOP_ROOT','VERL_ROOT','DT_ENVIRONMENT_JSON')}
 report['jobs'].append(row)
report['physical']=subprocess.check_output(['mx-smi'],text=True)
report['host_available_bytes']=psutil.virtual_memory().available
report['cgroup_bytes']=int(pathlib.Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
print(json.dumps(report))
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',ROOT)
p=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
text=p.stdout.decode('utf8','replace')
if p.returncode:
 print(text);raise SystemExit(p.returncode)
value=json.loads(text)
directory=AUDIT/'phase-observation-20261004';directory.mkdir(exist_ok=True)
path=directory/f'prefix-context-{int(value["observed_unix"])}.json';path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
for row in value['jobs']:
 row['logs']=[dict(path=l['path'],retry_match_count=l['retry_match_count'],retry_matches=l['retry_matches'][-2:],last_completed_metrics=l['last_completed_metrics']) for l in row['logs']]
print(json.dumps(dict(local_receipt=str(path),**value),ensure_ascii=False,indent=2))
