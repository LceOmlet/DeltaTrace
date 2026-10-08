source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/native-context-readouts-20261009-v1'
import psutil,json,time,subprocess
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes());rows=[]
try:
 p=psutil.Process(launch['pid']);same=p.create_time()==launch['birth']
 if same:
  for child in [p,*p.children(recursive=True)]:
   try:rows.append(dict(pid=child.pid,birth=child.create_time(),status=child.status(),cpu=child.cpu_times()._asdict(),memory=child.memory_full_info()._asdict(),cmd=child.cmdline()))
   except psutil.Error:pass
except psutil.NoSuchProcess:same=False
r=dict(unix=time.time(),driver_alive=same and bool(rows),processes=rows,physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,host=psutil.virtual_memory()._asdict(),log=(out/'driver.log').read_text()[-10000:])
if (out/'result.json').exists():r['result']=json.loads((out/'result.json').read_bytes())
print(json.dumps(r))

PY
