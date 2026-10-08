source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-gdn-collection-textcraft-20261009-v1'
import json,psutil,subprocess,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
same=False;processes=[]
try:
 p=psutil.Process(launch['pid']);same=p.create_time()==launch['birth']
 if same:
  for child in [p,*p.children(recursive=True)]:
   try:processes.append(dict(pid=child.pid,birth=child.create_time(),status=child.status(),cpu=child.cpu_times()._asdict(),memory=child.memory_full_info()._asdict(),cmd=child.cmdline()))
   except psutil.Error:pass
except psutil.NoSuchProcess:pass
records={};logs={}
for rank in range(2):
 p=out/'results'/('rank'+str(rank)+'.json')
 if p.exists():records[str(rank)]=json.loads(p.read_bytes())
 p=out/'results'/('rank'+str(rank)+'-phases.jsonl')
 if p.exists():logs[str(rank)]=p.read_text()[-7000:]
for p in (out/'results').glob('gdn-context-pid*.jsonl'):
 logs[p.name]=p.read_text()[-9000:]
r=dict(unix=time.time(),launch=launch,driver_alive=same,processes=processes,records=records,phase_logs=logs,
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
 host=psutil.virtual_memory()._asdict(),log=(out/'driver.log').read_text()[-10000:])
print(json.dumps(r))

PY
