source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'

import hashlib,json,psutil,subprocess,time
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
f=r/'runs/textcraft-formal-stable-20261009-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((f/'source.json').read_bytes());e=s['environment']
q=json.loads(Path(e['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
dt=Path(e['DT_ROOT'])
files=[dt/'profiles/official.py',dt/'profiles/qwen35_gdn_symmetric.py',
 dt/'clean/qwen35/qwen35_clean_runner.py',
 dt/'clean/qwen35/qwen35_gdn_finite.py',
 dt/'clean/qwen35/qwen35_dense_finite_runner.py',
 f/'entry/deltatrace_rollout.py']
records=[]
for p in files:
 v=dict(path=str(p),exists=p.exists())
 if p.exists():v.update(resolved=str(p.resolve()),sha256=sha(p),text=p.read_text())
 records.append(v)
workers=[]
for pid,birth in [(982372,1791553809.84),(987808,1791553850.),(989860,1791553867.51)]:
 try:
  p=psutil.Process(pid);workers.append(dict(pid=pid,expected_birth=birth,birth=p.create_time(),name=p.name()))
 except psutil.NoSuchProcess:workers.append(dict(pid=pid,exists=False))
result=dict(unix=time.time(),source_sha256=sha(f/'source.json'),
 owners=records,workers=workers,config=q,
 environment_paths={k:v for k,v in e.items() if k in ['DT_ROOT','DT_OFFICIAL_ROOT','DT_ENVIRONMENT_JSON','VENV_PYTHON','PYTHONPATH']},
 pythonpath=s.get('pythonpath'),startup=s.get('startup_options'),
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
 host_available=psutil.virtual_memory().available,
 model_calls=0,DT_calls=0,optimizer_calls=0,production_changes=0)
print(json.dumps(result))

PY
