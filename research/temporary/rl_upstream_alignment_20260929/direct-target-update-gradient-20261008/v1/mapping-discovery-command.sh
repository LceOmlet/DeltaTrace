set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json, torch,sys,hashlib
from pathlib import Path
root=Path("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922")
directory=root/"receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt"
r={}
for rank in (0,1):
 d=torch.load(directory/f"rank{rank}-pre-update.pt",map_location="cpu",weights_only=False)
 a=d["non_tensors"]["dt_direct_target_artifact"][0]
 r[rank]={"tensors":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in d["tensors"].items()},"non_tensors":{k:{"type":str(type(v)),"len":len(v),"first_type":str(type(v[0]))} for k,v in d["non_tensors"].items()},"artifact_keys":list(a),"artifact_fields":{k:{"type":str(type(v)),"len":len(v) if hasattr(v,"__len__") else None,"shape":list(v.shape) if hasattr(v,"shape") else None,"preview":str(v)[:180]} for k,v in a.items()},"provenance":d["provenance"],"raw_readout_path":d["raw_readout_path"]}
print(json.dumps(r))
assert not torch.cuda.is_initialized()

PY
