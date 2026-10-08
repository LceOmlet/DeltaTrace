source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-attention-input-textcraft-20261009-v1'
import hashlib,json,psutil,subprocess,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes());alive=False
try:alive=psutil.Process(launch['pid']).create_time()==launch['birth']
except psutil.Error:pass
names=['launch.json','preparation.json','driver.log','observer-source.tar','suboperation-protocol.json','passive_attention_gate.py','inspect_layer_collection.py','passive_suboperations.py','results/rank0.json','results/rank1.json','results/rank0-phases.jsonl','results/rank1-phases.jsonl']
for rank in (0,1):assert json.loads((out/'results'/f'rank{rank}.json').read_bytes())['phase']=='complete'
files=[dict(remote=str(out/n),bytes=(out/n).stat().st_size,sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in names]
value=dict(unix=time.time(),driver_alive=alive,files=files,physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,host=psutil.virtual_memory()._asdict())
print(json.dumps(value))

PY
