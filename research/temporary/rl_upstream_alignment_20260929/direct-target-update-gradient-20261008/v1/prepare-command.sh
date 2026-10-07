set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,hashlib,time,psutil,subprocess
from pathlib import Path
import torch
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');directory=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
r=dict(unix=time.time(),driver_same_birth=psutil.Process(2833207).create_time()==1791370325.16,files=[])
for rank in (0,1):
 p=directory/('rank'+str(rank)+'-pre-update.pt');d=torch.load(p,map_location='cpu',weights_only=False)
 item=dict(rank=rank,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),type=str(type(d)))
 if isinstance(d,dict):
  item['keys']=list(d);item['fields']={k:dict(type=str(type(v)),shape=list(v.shape) if hasattr(v,'shape') else None,keys=list(v) if isinstance(v,dict) else None) for k,v in d.items()}
 r['files'].append(item)
r['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
r['text_release_present']=[(directory/('rank'+str(i)+'-release-update')).exists() for i in (0,1)]
r['cuda_initialized']=torch.cuda.is_initialized()
assert not r['cuda_initialized']
print(json.dumps(r))

PY
