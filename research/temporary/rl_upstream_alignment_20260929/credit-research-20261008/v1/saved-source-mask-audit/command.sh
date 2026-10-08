source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/saved-source-mask-audit-20261009-v1'
SHA='5554113547025398e3306cc3909c35b1ba303808212d8f43ad9c889fb840191e'
import hashlib,subprocess,os
from pathlib import Path
p=Path(OUT)/'check_saved_source_masks.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
subprocess.run([os.environ['VENV_PYTHON'],str(p),'--root',ROOT,'--plan',str(Path(OUT)/'layer-collection-inputs.json'),'--output',str(Path(OUT)/'result.json')],check=True)

PY
