"""Read actual native AdamW/Adam source for the saved-moment analytic readout."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import importlib
import torch

OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
adam=importlib.import_module('torch.optim.adam')
functions=[torch.optim.AdamW.step,adam._single_tensor_adam,adam._multi_tensor_adam]
rows=[]
for function in functions:
    function=inspect.unwrap(function);path=Path(inspect.getsourcefile(function))
    source,start=inspect.getsourcelines(function)
    selected=set()
    for i,line in enumerate(source):
        if any(word in line for word in ('bias_correction','step_size','denom','addcdiv_','decoupled_weight_decay','state_steps')):
            selected.update(range(max(0,i-1),min(len(source),i+2)))
    rows.append(dict(function=function.__module__+'.'+function.__qualname__,path=str(path),
                     SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     relevant_source=[dict(line=start+i,text=source[i].rstrip()) for i in sorted(selected)]))
report=dict(unix=time.time(),scope=__doc__,sources=rows,optimizer_calls=0,model_calls=0,
            interpretation='The constant-lr, non-AMSGrad, non-maximize saved groups use the recorded bias corrections and denominator in both native single/foreach branches. Statistical readout is FP64 algebra, not a bitwise replay or new optimizer.')
assert not torch.cuda.is_initialized()
dest=OUT/'native-adaptive-update-owner-v45.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
