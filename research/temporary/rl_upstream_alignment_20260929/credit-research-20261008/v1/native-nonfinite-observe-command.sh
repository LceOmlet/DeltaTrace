source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

import json,time,psutil,subprocess
from pathlib import Path
out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-nonfinite-20261008-v1')
launch=json.loads((out/'launch.json').read_bytes())
result=dict(observed_unix=time.time(),launch=launch,processes=[],ranks=[])
try:
 p=psutil.Process(launch['pid']);assert p.create_time()==launch['birth']
 for q in [p,*p.children(recursive=True)]:
  try: result['processes'].append(dict(pid=q.pid,birth=q.create_time(),status=q.status(),name=q.name(),pss_bytes=q.memory_full_info().pss))
  except(psutil.NoSuchProcess,psutil.AccessDenied):pass
except psutil.NoSuchProcess:pass
for p in out.glob('rank[01].json'):result['ranks'].append(json.loads(p.read_bytes()))
p=out/'phase.json';result['phase']=json.loads(p.read_bytes()) if p.exists() else None
result['log_tail']=(out/'driver.log').read_text(errors='replace')[-14000:]
result['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout
print(json.dumps(result))

PY
