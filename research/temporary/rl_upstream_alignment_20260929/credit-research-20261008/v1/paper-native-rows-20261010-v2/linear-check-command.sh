source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
REMOTE='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/paper-native-rows-20261010-v2'
PYTHON='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'

import hashlib,json,subprocess,time
from pathlib import Path
out=Path(REMOTE)
assert (out/'completed.json').exists()
script=out/'check_native_linear_roundoff.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='ac22648a404497ecc785ca54685dabe9565ed864363152927e64410ae22c8b2d'
argv=[PYTHON,str(script),'--folder',str(out),'--output',str(out/'linear-roundoff.json')]
start=time.time()
run=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=55)
receipt=dict(unix=start,completed_unix=time.time(),argv=argv,stdout=run.stdout,stderr=run.stderr,
 script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),GPU_calls=0,thresholds_changed=False)
(out/'linear-check-invocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

PY
