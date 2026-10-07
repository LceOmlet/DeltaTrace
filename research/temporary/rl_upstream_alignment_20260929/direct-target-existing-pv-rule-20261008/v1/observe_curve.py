"""Read the exact original-author comparison job; no runtime mutation."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import json,psutil,subprocess,time
from pathlib import Path
root=Path(ROOT);out=root/'receipts/direct-target-existing-pv-rule-20261008-v1-curves'
a=json.loads((out/'launch.json').read_bytes())
p=psutil.Process(a['pid']) if psutil.pid_exists(a['pid']) else None
r=dict(unix=time.time(),driver=dict(pid=a['pid'],birth=a['birth'],same_birth=p is not None and p.create_time()==a['birth']),completed=(out/'results/completed.json').exists(),ranks=[])
for rank in (0,1):
 f=out/'results'/('rank'+str(rank)+'.json')
 if f.exists():
  d=json.loads(f.read_bytes())
  entry={k:d[k] for k in ('rank','pid','birth','phase','active_view','completed_points','last_point','traceback','native_forward_calls') if k in d}
  entry['views']={key:dict(author_return=value['author_return'],points=len(value['score_points']),seconds=sum(x['seconds'] for x in value['score_points'])) for key,value in d.get('views',{}).items()}
  if psutil.pid_exists(d['pid']):
   w=psutil.Process(d['pid'])
   if w.create_time()==d['birth']:entry['current_pss_bytes']=w.memory_full_info().pss
  r['ranks'].append(entry)
r['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
for name in ('memory.usage_in_bytes','memory.stat'):r[name]=(Path('/sys/fs/cgroup/memory')/name).read_text()
r['textcraft_same_birth']=psutil.Process(2833207).create_time()==1791370325.16
r['textcraft_release_present']=[(root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1)]
if not r['ranks'] or not r['driver']['same_birth']:r['driver_log_tail']=(out/'driver.log').read_text(errors='replace')[-6000:]
print(json.dumps(r))
'''.replace('ROOT', repr(transport.ROOT))
shell = 'set -eu\nsource ' + transport.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
r = subprocess.run(transport.SSH + ['bash', '-s'], input=shell.encode(), capture_output=True)
if r.returncode:
    print(r.stderr.decode(errors='replace'))
r.check_returncode()
record = json.loads(r.stdout)
(HERE / ('curve-observation-' + str(int(record['unix'])) + '.json')).write_text(json.dumps(record, indent=2) + '\n')
# Exact positions remain in the saved receipt; keep the live report compact.
for entry in record['ranks']:
    if 'last_point' in entry:
        entry['last_point'] = {k:v for k,v in entry['last_point'].items() if k != 'changed_input_positions'}
physical = record.pop('physical_mx_smi')
record['physical_memory_mib'] = {}
gpu = None
for line in physical.splitlines():
    board = re.match(r'^\|\s*(\d+)\s+MetaX\s', line)
    if board: gpu = board[1]
    memory = re.search(r'(\d+)/(\d+) MiB', line)
    if memory: record['physical_memory_mib'][gpu] = int(memory[1])
record['cgroup_bytes'] = int(record.pop('memory.usage_in_bytes'))
record.pop('memory.stat')
print(json.dumps(record))
