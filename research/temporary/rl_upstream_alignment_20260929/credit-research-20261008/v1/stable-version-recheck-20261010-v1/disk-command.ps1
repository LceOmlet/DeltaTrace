@'
import sys,subprocess
sys.path.insert(0,'research/temporary/rl_upstream_alignment_20260929')
import stage_environment_entry as t
s="""source %s/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import json,hashlib,time,psutil
from pathlib import Path
r=Path(%r);f=r/'runs/textcraft-formal-stable-20261009-v1'
m=json.loads((f/'source.json').read_bytes());checks=[]
for name,v in m['actual_CPU_imports'].items():
 p=Path(v['path']); checks.append(dict(name=name,path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),expected=v['sha256']))
g=Path(m['dt_root'])/'clean/qwen35/qwen35_gdn_finite.py'
checks.append(dict(name='qwen35_gdn_finite',path=str(g),resolved=str(g.resolve()),sha256=hashlib.sha256(g.read_bytes()).hexdigest(),expected='7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656'))
assert all(x['sha256']==x['expected'] for x in checks),[x for x in checks if x['sha256']!=x['expected']]
record=dict(unix=time.time(),formal_pid=982372,formal_birth=psutil.Process(982372).create_time(),numerical_runtime=m['numerical_runtime'],startup_options=m['startup_options'],lora_rank=m['lora_rank'],lora_alpha=m['lora_alpha'],fresh_base_model=m['fresh_base_model'],resume_mode=m['resume_mode'],checkpoint_restore_requested=m['checkpoint_restore_requested'],checks=checks,runtime_overrides=[str(p) for p in (f/'runtime-overrides').glob('*.json')])
out=r/'receipts/stable-version-recheck-20261010-v1'
(out/'disk-and-config.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record,indent=2))
q=out/'worker-runtime.json'
if q.exists():
 a=json.loads(q.read_bytes());print('RPC_STATUS',json.dumps(dict(complete=a.get('complete'),started_unix=a.get('started_unix'),completed_unix=a.get('completed_unix'))))
else:print('RPC_STATUS pending_initialization')
PY
"""%(t.ENTRY,t.ROOT)
r=subprocess.run(t.SSH+['bash','-s'],input=s.encode(),capture_output=True,timeout=35)
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));r.check_returncode()
'@ | C:/Users/Administrator/miniconda3/python.exe -X utf8 -
