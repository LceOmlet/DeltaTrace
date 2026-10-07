"""Read phase/PSS/physical memory for the exact diagnostic process handle."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
parser=argparse.ArgumentParser();parser.add_argument('task',choices=('textcraft','appworld'));args=parser.parse_args()
code=r'''
from pathlib import Path
import json,psutil,subprocess,time
root=Path(@ROOT@);out=root/'receipts/direct-target-credit-sample-20261007-v1';task=@TASK@
a=json.loads((out/(task+'-launch.json')).read_bytes());p=psutil.Process(a['pid']) if psutil.pid_exists(a['pid']) else None
r=dict(unix=time.time(),task=task,driver=dict(pid=a['pid'],birth=a['birth'],same_birth=p is not None and p.create_time()==a['birth']),completed=(out/('results-'+task)/'completed.json').exists(),ranks=[])
for i in (0,1):
 f=out/('results-'+task)/(task+'-rank'+str(i)+'.json')
 if f.exists():
  d=json.loads(f.read_bytes());v={k:d[k] for k in ('rank','pid','birth','phase','active_mode','traceback') if k in d};v['modes']={k:dict(seconds=vv['seconds'],factual_equal=vv['factual_target_logp_equal_to_identity'],rows=[dict(uid=rr['traj_uid'],d=rr['single_delete_d'],A=rr.get('native_endpoint_expected_A_FP32')) for rr in vv['rows']]) for k,vv in d.get('modes',{}).items()};r['ranks'].append(v)
  f=out/('results-'+task)/(task+'-rank'+str(i)+'-phases.jsonl')
  if f.exists():r['ranks'][-1]['last_phase']=json.loads(f.read_text().splitlines()[-1])
r['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
for name in ('memory.usage_in_bytes','memory.stat'):r[name]=(Path('/sys/fs/cgroup/memory')/name).read_text()
r['textcraft_same_birth']=psutil.Process(2833207).create_time()==1791370325.16
r['textcraft_release_present']=[(root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1)]
if not r['ranks'] or not r['driver']['same_birth']:r['log_tail']=(out/(task+'-driver.log')).read_text(errors='replace')[-5000:]
print(json.dumps(r))
'''.replace('@ROOT@',repr(transport.ROOT)).replace('@TASK@',repr(args.task))
shell='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
if r.returncode:print(r.stderr.decode(errors='replace'))
r.check_returncode();record=json.loads(r.stdout)
(HERE/(args.task+'-observation-'+str(int(record['unix']))+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
