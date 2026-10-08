source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
import hashlib,json
from pathlib import Path
out={}
for task in ('textcraft','appworld'):
 p=Path(ROOT)/f'runs/direct-target-prefix-runtime-20261007-v1/{task}/{task}-dt/source.json'
 s=json.loads(p.read_bytes());out[task]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),keys=list(s),pythonpath=s.get('pythonpath'),imports=s.get('actual_CPU_imports'))
print(json.dumps(out))

PY
