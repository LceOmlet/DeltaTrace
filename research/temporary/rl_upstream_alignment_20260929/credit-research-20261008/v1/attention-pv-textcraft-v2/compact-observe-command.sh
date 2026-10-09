source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-attention-pv-textcraft-20261009-v2'

import json,psutil,time,subprocess,os
from pathlib import Path
out=Path(OUT);l=json.loads((out/'launch.json').read_bytes())
try:alive=psutil.Process(l['pid']).create_time()==l['birth']
except psutil.Error:alive=False
ranks={}
for rank in (0,1):
 p=out/'results'/f'rank{rank}.json'
 if p.exists():
  v=json.loads(p.read_bytes());points=[p for b in v['batches'] for p in b['points']]
  ranks[str(rank)]=dict(phase=v['phase'],elapsed=v['elapsed_seconds'],points=len(points),operations=v['operations'],last_phase=(out/'results'/f'rank{rank}-phases.jsonl').read_text().splitlines()[-1],PV_points=sum('final_FA_PV_ledger' in p for p in points))
print(json.dumps(dict(unix=time.time(),driver_alive=alive,ranks=ranks,physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,hostavailable=psutil.virtual_memory().available,disk_free=os.statvfs(out).f_bavail*os.statvfs(out).f_frsize)))

PY
