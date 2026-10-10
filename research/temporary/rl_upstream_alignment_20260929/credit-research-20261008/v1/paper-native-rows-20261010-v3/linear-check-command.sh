source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
REMOTE='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/paper-native-rows-20261010-v3'
PYTHON='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'

import hashlib,json,subprocess,time
from pathlib import Path
out=Path(REMOTE)
assert (out/'completed.json').exists()
script=out/'check_native_linear_roundoff.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='21bdd3b3205b6b7a6d789d80edffcea00d614cd5f00ba1ae9f2d2f8276f89ce5'
argv=[PYTHON,str(script),'--folder',str(out),'--output',str(out/'linear-roundoff.json')]
start=time.time()
run=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=55)
receipt=dict(unix=start,completed_unix=time.time(),argv=argv,stdout=run.stdout,stderr=run.stderr,
 script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),GPU_calls=0,thresholds_changed=False)
(out/'linear-check-invocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

PY
