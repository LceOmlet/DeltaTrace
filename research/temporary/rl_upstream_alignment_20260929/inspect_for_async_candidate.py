"""Read current original trainer phases while preparing the transport seam."""
from stage_environment_entry import remote,ROOT,ENTRY

remote(fr'''source {ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,psutil,time,pathlib,re,subprocess
root=pathlib.Path('{ROOT}');active=json.loads((root/'active-training.json').read_text());rows=[]
for job in active['jobs']:
 driver=psutil.Process(job['pid']); expected=job.get('observed_process_created_unix',driver.create_time())
 assert abs(driver.create_time()-expected)<.02
 row=dict(task=job['task'],pid=driver.pid,created_unix=driver.create_time(),pss_bytes=0,logs=[])
 seen=set()
 for p in [driver]+driver.children(recursive=True):
  try:
   row['pss_bytes']+=p.memory_full_info().pss
   for f in p.open_files():
    if f.path in seen or '/worker-' not in f.path or not f.path.endswith('.out'):continue
    seen.add(f.path)
    if pathlib.Path(f.path).stat().st_size>200*1024*1024:continue
    matches=[]
    with open(f.path,errors='replace') as stream:
     for line in stream:
      if re.match(r'^(step:|\[loop_transport\]|\[loop_trajectory\]|\[dt_rollout\]|\[owner_trajectory\])',line):matches.append(line.strip()[:1100])
    if matches:row['logs'].append(dict(path=f.path,latest=matches[-2:]))
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 marker=pathlib.Path(job['checkpoints'])/'latest_checkpointed_iteration.txt'
 row['checkpoint_marker']=marker.read_text().strip() if marker.exists() else None
 rows.append(row)
report=dict(observed_unix=time.time(),jobs=rows)
cgroup=pathlib.Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
if cgroup.exists():report['cgroup_memory_bytes']=int(cgroup.read_text())
destination=root/'receipts/owner-b8-dispatch-20260930/appworld-batch-coalescing-20261002/native-profiler/current-phase.json'
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
''')
