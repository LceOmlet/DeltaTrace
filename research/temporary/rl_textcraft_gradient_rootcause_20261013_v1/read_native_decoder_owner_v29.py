"""Read the actual native last decoder residual/attention owners; no model call."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as owner

OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
path=Path(inspect.getsourcefile(owner))
assert hashlib.sha256(path.read_bytes()).hexdigest()=='59f9c339e3c01672b67ba09b0e56e2e960ba28c6c3d9975faadf00d15ee0b4c8'
names=('Qwen3_5Attention','Qwen3_5DecoderLayer','Qwen3_5MLP','Qwen3_5RMSNorm')
report=dict(unix=time.time(),scope=__doc__,owner_path=str(path),owner_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
    objects={name:inspect.getsource(getattr(owner,name)) for name in names},GPU_model_calls=0,training_changed=False)
assert not torch.cuda.is_initialized()
dest=OUT/'native-decoder-owner-v29.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
