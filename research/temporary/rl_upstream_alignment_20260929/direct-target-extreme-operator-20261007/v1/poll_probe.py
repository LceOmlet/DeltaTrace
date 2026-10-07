"""Read the exact launched handle, phase workload and physical resources."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
code=r'''
from pathlib import Path
import json,psutil,subprocess,time
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=root/'receipts/direct-target-extreme-operator-20261007-v1'
a=json.loads((out/'launch.json').read_bytes());p=psutil.Process(a['pid']) if psutil.pid_exists(a['pid']) else None
r=dict(unix=time.time(),driver=dict(pid=a['pid'],birth=a['birth'],same_birth=p is not None and p.create_time()==a['birth']),completed=(out/'results/completed.json').exists(),ranks=[])
for i in (0,1):
 f=out/'results'/('rank'+str(i)+'.json')
 if f.exists():
  d=json.loads(f.read_bytes());v={k:d[k] for k in ('rank','pid','birth','phase','active_mode','traceback') if k in d};v['modes']=list(d['phases']);v['cross_boundaries']=len(d.get('cross_boundary_contractions',{}));r['ranks'].append(v)
 f=out/'results'/('rank'+str(i)+'-phases.jsonl')
 if f.exists():r['ranks'][-1]['last_phase']=json.loads(f.read_text().splitlines()[-1])
r['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
for name in ('memory.usage_in_bytes','memory.stat'):r[name]=(Path('/sys/fs/cgroup/memory')/name).read_text()
r['textcraft_same_birth']=psutil.Process(2833207).create_time()==1791370325.16
r['textcraft_release_present']=[(root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1)]
if not r['ranks'] or not r['driver']['same_birth']:r['log_tail']=(out/'driver.log').read_text(errors='replace')[-5000:]
print(json.dumps(r))
'''
shell='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True,check=True)
record=json.loads(r.stdout);(HERE/('observation-'+str(int(record['unix']))+'.json')).write_text(json.dumps(record,indent=2)+'\n')
record.pop('physical_mx_smi');record.pop('memory.stat');print(json.dumps(record))
