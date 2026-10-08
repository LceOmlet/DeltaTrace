"""Read native launch/initialization seams and current task service; no training."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import hashlib,json,psutil,subprocess,time,urllib.request
from pathlib import Path
root=Path(ROOT)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
source=json.loads(source_path.read_bytes())
files=[]
for relative in ('verl/trainer/main_ppo.py','verl/trainer/ppo/ray_trainer.py','verl/single_controller/ray/base.py','verl/utils/model.py'):
 path=Path(source['verl_root'])/relative
 raw=path.read_bytes();files.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),source=raw.decode()))
service=Path(source['current_service_receipt']['path'])
service_info=json.loads(service.read_bytes())
try:
 with urllib.request.urlopen('http://127.0.0.1:36005/docs',timeout=5) as response:
  health=dict(status=response.status,bytes=len(response.read()))
except Exception as error: health=dict(error=repr(error))
processes=[]
for p in psutil.process_iter(['pid','create_time','cmdline']):
 try:
  cmd=' '.join(p.info['cmdline'] or [])
  if str(root) in cmd and any(k in cmd for k in ('main_ppo','diagnose_','textcraft','ray')):
   processes.append(dict(pid=p.pid,birth=p.info['create_time'],command=cmd[:800]))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
print(json.dumps(dict(unix=time.time(),files=files,service=service_info,health=health,processes=processes,
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout)))
'''
script = 'source ' + transport.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\nROOT=' + repr(transport.ROOT) + '\n' + code + '\nPY\n'
result = subprocess.run(transport.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=60)
(HERE / 'read-native-first-iteration.stderr.txt').write_bytes(result.stderr)
if result.returncode:
    print(result.stderr.decode(errors='replace'))
result.check_returncode()
value = json.loads(result.stdout)
(HERE / 'native-first-iteration-sources.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k:v for k,v in value.items() if k!='files'}))
