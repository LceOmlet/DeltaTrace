source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-attention-pv-textcraft-20261009-v1'

import hashlib,json,psutil,subprocess,tarfile,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
try:alive=psutil.Process(launch['pid']).create_time()==launch['birth']
except psutil.Error:alive=False
assert not alive
names=['launch.json','preparation.json','driver.log','observer-source.tar','suboperation-protocol.json',*launch['scripts'],'results/rank0.json','results/rank1.json','results/rank0-phases.jsonl','results/rank1-phases.jsonl']
names=list(dict.fromkeys(names));names=[n for n in names if (out/n).exists()]
archive=out/'diagnostic-debug.tgz'
with tarfile.open(archive,'x:gz') as stream:
 for name in names:stream.add(out/name,arcname=name)
files=[dict(remote=str(out/n),name=n,bytes=(out/n).stat().st_size,sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in names]
value=dict(unix=time.time(),driver_alive=alive,files=files,archive=dict(remote=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),status='Failed observer assertion; no completed PV measurement',physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,host=psutil.virtual_memory()._asdict())
print(json.dumps(value))

PY
