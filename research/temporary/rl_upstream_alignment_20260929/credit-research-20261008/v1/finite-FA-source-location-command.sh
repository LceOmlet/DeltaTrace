source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import hashlib,json
from pathlib import Path
path=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/environment.json');raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()=='4ff007805297bfd5caddf7158f5b3198c42b41597951d7f226b0808f109b998e'
q=json.loads(raw)['qwen35'];print(json.dumps({k:v for k,v in q.items() if any(s in k.lower() for s in ('finite','extension','fa_'))}))

PY
