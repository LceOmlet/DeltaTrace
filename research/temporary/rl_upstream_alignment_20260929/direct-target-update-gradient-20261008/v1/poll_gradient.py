"""Read bounded diagnostic's original PID/birth, phases and resources."""
from pathlib import Path
import importlib.util,subprocess,json
HERE=Path(__file__).resolve().parent;AUDIT=HERE.parents[1]
s=importlib.util.spec_from_file_location('stage',AUDIT/'stage_environment_entry.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
remote=m.ROOT+'/receipts/direct-target-update-gradient-20261008-v1'
code=r"""
import json,psutil,time,subprocess
from pathlib import Path
out=Path(@OUT@);launch=json.loads((out/'gradient-launch.json').read_bytes());r=dict(unix=time.time(),launch=launch)
p=psutil.Process(launch['pid']) if psutil.pid_exists(launch['pid']) else None
same=p is not None and p.create_time()==launch['birth'];r['driver_same_birth']=same
r['processes']=[]
if same:
 for item in [p,*p.children(recursive=True)]:
  try:r['processes'].append(dict(pid=item.pid,birth=item.create_time(),status=item.status(),name=item.name(),pss_bytes=item.memory_full_info().pss))
  except(psutil.NoSuchProcess,psutil.AccessDenied):pass
for name in ['phase.json','completed.json','input-inspection.json','rank0-gradients.json','rank1-gradients.json']:
 path=out/name
 if path.exists():r[name]=json.loads(path.read_bytes())
log=out/'gradient-driver.log';r['driver_log_tail']=log.read_text(errors='replace')[-11000:] if log.exists() else ''
r['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
base=out.parents[1]/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt';r['text_releases']=[(base/f'rank{i}-release-update').exists() for i in (0,1)]
r['text_same_birth']=psutil.pid_exists(2833207) and psutil.Process(2833207).create_time()==1791370325.16
print(json.dumps(r))
""".replace('@OUT@',repr(remote))
script='set -eu\nsource '+m.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(m.SSH+['bash','-s'],input=script.encode(),capture_output=True)
if r.returncode:print(r.stderr.decode(errors='replace'));r.check_returncode()
d=json.loads(r.stdout);(HERE/f"gradient-observation-{int(d['unix'])}.json").write_bytes(r.stdout)
print(json.dumps({k:v for k,v in d.items() if k not in ['physical','driver_log_tail','input-inspection.json','rank0-gradients.json','rank1-gradients.json']},ensure_ascii=False));print(d['driver_log_tail'][-5000:]);print(d['physical'])
