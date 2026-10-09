set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
cd /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-tail-probability-sample-20261009-v1
tar -xf native-source.tar
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,subprocess,time
from pathlib import Path
out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-tail-probability-sample-20261009-v1')
assert not (out/"native-controller-launch.json").exists(), "Do not duplicate collection"
with (out/"native-controller.log").open("xb") as log:
 p=subprocess.Popen([os.environ["VENV_PYTHON"],str(out/"run_tail_probability_chunks.py"),str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),prepared_sha256=hashlib.sha256((out/"native-prepared.json").read_bytes()).hexdigest())
(out/"native-controller-launch.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
PY
