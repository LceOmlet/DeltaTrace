source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import json,os,psutil,subprocess,time,hashlib
from pathlib import Path
out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-FA-source-reference-20261009-v1'); launch=json.loads((out/'launch.json').read_bytes());pid=launch['pid']
try:p=psutil.Process(pid);alive=abs(p.create_time()-launch['birth'])<.1 and p.status()!=psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:alive=False
r=json.loads((out/'result.json').read_bytes()) if (out/'result.json').exists() else {}
files={name:dict(bytes=(out/name).stat().st_size,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ['result.json','result.phases.jsonl','driver.log','launch.json'] if (out/name).exists()}
print(json.dumps(dict(unix=time.time(),alive=alive,phase=r.get('phase'),points=len(r.get('points',[])),elapsed=r.get('elapsed_seconds'),operations=r.get('operations'),PSS=r.get('sampled_PSS_peak_bytes'),GPU_allocated=r.get('GPU_peak_allocated_bytes'),GPU_reserved=r.get('GPU_peak_reserved_bytes'),traceback=r.get('traceback'),log_tail=(out/'driver.log').read_text(errors='replace')[-2400:],physical=subprocess.check_output(['mx-smi'],text=True),files=files)))

PY
