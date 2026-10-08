source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
FILES={'textcraft': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank0-readout-native-batch-1.pt', 'sha256': '3191adac4bb2a13c0ea6cc1cc18bffbe2577c969b14696e4b6e313d04f43ea8b', 'bytes': 2320803}, 'appworld': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-prefix-runtime-20261007-v1/appworld-first-dt/rank0-readout-native-batch-1.pt', 'sha256': 'b62f251d65e0b194a9749c54aa9dd3e6b5b4a0e47119f243e24226fd6479241d', 'bytes': 4900067}}
import hashlib,json,torch
from pathlib import Path
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
out={}
for task,ref in FILES.items():
 p=Path(ref['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
 x=torch.load(p,map_location='cpu',weights_only=False)
 def structure(v):
  if isinstance(v,torch.Tensor):return dict(shape=list(v.shape),dtype=str(v.dtype),device=str(v.device))
  if isinstance(v,dict):return {k:structure(a) for k,a in v.items()}
  if isinstance(v,list):return dict(length=len(v),first=structure(v[0]) if v else None)
  if isinstance(v,(str,int,float,bool)) or v is None:return v
  return str(type(v))
 out[task]=structure(x)
print(json.dumps(out))
assert not torch.cuda.is_initialized()

PY
