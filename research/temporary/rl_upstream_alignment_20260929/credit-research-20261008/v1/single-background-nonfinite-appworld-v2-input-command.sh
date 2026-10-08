source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-single-background-nonfinite-appworld-20261009-v2'
import torch,json,hashlib,time
from pathlib import Path
torch.set_num_threads(8);out=Path(OUT);p=out/'results/rank0-first-nonfinite.pt';x=torch.load(p,map_location='cpu',weights_only=False)
result={'unix':time.time(),'artifact_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'callback':x['callback'],'layer':x['layer'],'inputs':[]}
def visit(v,key):
 if isinstance(v,torch.Tensor):
  valid=torch.isfinite(v);bad=(~valid).nonzero()
  result['inputs'].append(dict(key=key,shape=list(v.shape),dtype=str(v.dtype),nonfinite=int(len(bad)),finite_maxabs=float(v[valid].abs().max()),first_nonfinite=bad[:8].tolist(),values=[v[tuple(i)].item() for i in bad[:8].tolist()]))
 elif isinstance(v,(tuple,list)):
  for i,item in enumerate(v):visit(item,key+'.'+str(i))
 elif isinstance(v,dict):
  for name,item in v.items():visit(item,key+'.'+str(name))
visit(x['args'],'args');visit(x['kwargs'],'kwargs')
p=out/'results/rank0-first-nonfinite-inputs.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

PY
