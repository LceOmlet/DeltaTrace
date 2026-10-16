"""Shape-only cost comparison using the original finite MLP and stored weights.

No model, PPO, DT propagation or replacement MLP is implemented. Explicit
component fixtures compare identical leading rows of the original operator
at partial/full2048-token chunk shapes. Differences are measured, not accepted
against an invented tolerance. Nothing is deployed by this probe.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-dt-cost-breakdown-20261016-v1'
text=subprocess.check_output(['mx-smi'],text=True)
assert not any(re.match(r'\|\s*6\s+\d+\s+\S+',line) for line in text.splitlines()),'GPU6 occupied; no process stopped'
import torch
from safetensors import safe_open
import qwen35_decoder_finite as owner
torch.cuda.set_device(0)
env=json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
op=owner.FiniteBoundaryOps(True,dynamic_shapes=env['dt_dynamic_shapes'],compiler_options=env['dt_compiler_options'],mlp_token_chunk_size=2048).mlp
model=Path(os.environ['MODEL_PATH']);index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
weights=[];bindings=[]
for kind in ['down_proj','up_proj','gate_proj']:
    key=next(k for k in index if k.endswith(f'layers.0.mlp.{kind}.weight'))
    with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:w=f.get_tensor(key).cuda()
    weights.append((w,torch.zeros_like(w)))
    bindings.append(dict(key=key,path=str(model/index[key]),shape=list(w.shape),dtype=str(w.dtype)))
state=dict(started_unix=time.time(),original_owner_path=owner.__file__,original_owner_SHA256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),checks=[],deployed=False,model_calls=0,optimizer_calls=0)
output=AUDIT/'original-MLP-layout-cost.json'
for length in [6906,10065,16250]:
    paired=[]
    for index in range(3):
        value=torch.empty((8,length,12288),dtype=torch.bfloat16,device='cpu' if index==0 else 'cuda',pin_memory=index==0)
        value.fill_(0.01);paired.append(value)
    args=[view for value in paired for view in [value[0::2],value[1::2]]]
    args.append(torch.full((4,length,4096),1e-4,device='cuda'))
    costs=[]
    for repeat in range(4):
        torch.cuda.synchronize();tick=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        result=op(*args,*weights);b.record();torch.cuda.synchronize()
        costs.append(dict(wall=time.perf_counter()-tick,stream=a.elapsed_time(b)/1000));del result
    row=dict(length=length,strides=[list(v.stride()) for v in args],devices=[str(v.device) for v in args],costs=costs)
    state['checks'].append(row);output.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(row),flush=True)
    del args,paired,value
state.update(completed_unix=time.time(),status='completed');output.write_text(json.dumps(state,indent=2)+'\n')
