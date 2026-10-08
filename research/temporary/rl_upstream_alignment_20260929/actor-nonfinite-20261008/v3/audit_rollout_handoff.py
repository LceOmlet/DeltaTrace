"""Read incident-bound rollout/actor lifecycle sources; no model or run."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import hashlib,json,psutil,subprocess,time
from pathlib import Path
root=Path(ROOT)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
source=json.loads(source_path.read_bytes())
verl=Path(source['verl_root'])
paths=[verl/'verl/workers/fsdp_workers.py',verl/'verl/workers/sharding_manager/fsdp_vllm.py']
paths.extend(sorted((verl/'verl/workers/rollout/vllm_rollout').glob('*.py')))
paths.extend(sorted((verl/'verl/utils/peft_utils.py').parent.glob('*peft*')))
files=[]
for p in paths:
 if p.is_file():
  raw=p.read_bytes()
  files.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),
   launch_expected_sha256=source['source_bindings'].get(str(p)),source=raw.decode()))
processes=[]
for p in psutil.process_iter(['pid','create_time','cmdline']):
 try:
  cmd=' '.join(p.info['cmdline'] or [])
  if str(root) in cmd and any(x in cmd for x in ('python','timeout','ray')):
   processes.append(dict(pid=p.pid,birth=p.info['create_time'],command=cmd[:700]))
 except (psutil.NoSuchProcess,psutil.AccessDenied): pass
print(json.dumps(dict(unix=time.time(),source_path=str(source_path),
 source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),files=files,
 processes=processes,physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
 memory=dict(psutil.virtual_memory()._asdict()))))
'''
code = 'ROOT=' + repr(transport.ROOT) + '\n' + code
script = 'source ' + transport.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
result = subprocess.run(transport.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=60)
(HERE / 'audit-rollout-handoff.stderr.txt').write_bytes(result.stderr)
result.check_returncode()
value = json.loads(result.stdout)
(HERE / 'rollout-handoff-sources.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(unix=value['unix'],files=[{k:v for k,v in f.items() if k!='source'} for f in value['files']],
 processes=value['processes'],physical=value['physical'],memory=value['memory'])))
