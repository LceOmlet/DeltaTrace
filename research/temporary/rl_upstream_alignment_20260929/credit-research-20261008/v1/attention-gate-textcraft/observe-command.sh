source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-attention-gate-textcraft-20261009-v1'
import json,psutil,subprocess,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes());alive=False
try:alive=psutil.Process(launch['pid']).create_time()==launch['birth']
except psutil.Error:pass
records={};logs={}
for rank in range(2):
 p=out/'results'/f'rank{rank}.json'
 if p.exists():records[str(rank)]=json.loads(p.read_bytes())
 p=out/'results'/f'rank{rank}-phases.jsonl'
 if p.exists():logs[str(rank)]=p.read_text()[-4000:]
value=dict(unix=time.time(),driver_alive=alive,records=records,phase_logs=logs,physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,host=psutil.virtual_memory()._asdict(),log=(out/'driver.log').read_text()[-7000:])
print(json.dumps(value))

PY
