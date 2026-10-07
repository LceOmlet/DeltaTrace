"""Read the specific launched PID and original author progress; never restart."""
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
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=root/'receipts/direct-target-action-author-curve-20261007-v1';a=json.loads((out/'launch.json').read_bytes())
p=psutil.Process(a['pid']) if psutil.pid_exists(a['pid']) else None
r=dict(unix=time.time(),driver=dict(pid=a['pid'],birth=a['birth'],same_birth=p is not None and p.create_time()==a['birth'],status=p.status() if p else None),ranks=[])
for rank in (0,1):
 f=out/'results'/('rank'+str(rank)+'.json')
 if f.exists():
  b=json.loads(f.read_bytes());v={k:b[k] for k in ('rank','pid','birth','phase','unix','active_view','completed_points','last_point','native_forward_calls','traceback') if k in b};v['views']={name:dict(points=len(val['score_points']),author_return=val['author_return']) for name,val in b['views'].items()};r['ranks'].append(v)
r['completed']=(out/'results/completed.json').exists()
r['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
for name in ('memory.usage_in_bytes','memory.stat'):
 f=Path('/sys/fs/cgroup/memory')/name
 if f.exists():r[name]=f.read_text()
r['textcraft_same_driver']=psutil.Process(2833207).create_time()==1791370325.16
r['textcraft_release_present']=[(root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(rank)+'-release-update')).exists() for rank in (0,1)]
r['log_tail']=(out/'driver.log').read_text(errors='replace')[-3500:] if not r['ranks'] or not p else None
print(json.dumps(r))
'''
shell='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
result.check_returncode();record=json.loads(result.stdout)
(HERE/('observation-'+str(int(record['unix']))+'.json')).write_text(json.dumps(record,indent=2)+'\n')
compact=dict(record);compact.pop('physical_mx_smi',None);compact.pop('memory.stat',None)
for rank in compact['ranks']:
 if 'last_point' in rank:
  rank['last_point']=dict(rank['last_point']);rank['last_point']['changed_input_positions_count']=len(rank['last_point'].pop('changed_input_positions'))
print(json.dumps(compact))
