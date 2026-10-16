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
generator=torch.Generator(device='cpu').manual_seed(20261016)
full=[torch.randn((4,2048,12288),generator=generator,dtype=torch.float32).to(torch.bfloat16).cuda() for _ in range(6)]
full.append((torch.randn((4,2048,4096),generator=generator,dtype=torch.float32)*1e-4).cuda())
state=dict(started_unix=time.time(),scope=__doc__,original_owner_path=owner.__file__,
    original_owner_SHA256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),weight_bindings=bindings,
    adapter_operand='Explicit zero-delta component fixture, not reconstructed training adapter state',checks=[],
    model_calls=0,optimizer_calls=0,deployed=False)
output=AUDIT/'original-MLP-shape-cost.json'
def save():output.write_text(json.dumps(state,indent=2)+'\n')
def call(args):
    torch.cuda.synchronize();tick=time.perf_counter();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
    result=op(*args,*weights);end.record();torch.cuda.synchronize()
    return result,dict(wall_seconds=time.perf_counter()-tick,stream_seconds=start.elapsed_time(end)/1000)
reference,cold=call(full);state['full2048_first_call']=cold;save()
for length in [762,1024,1344,1873,1914,2048]:
    args=[v[:,:length].contiguous() for v in full]
    result,initial=call(args)
    diff=(result-reference[:,:length]).float()
    record=dict(tokens_per_trajectory=length,flattened_GEMM_rows=4*length,first_call=initial,
        full2048_needed_rows_bitwise_equal=torch.equal(result,reference[:,:length]),
        maximum_absolute_difference=float(diff.abs().max()),
        relative_L2_against_full2048=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(reference[:,:length].float())))
    del diff,result
    costs=[]
    for repeat in range(4):
        order=[('partial',args),('full2048',full)] if repeat%2==0 else [('full2048',full),('partial',args)]
        for label,values in order:
            result,cost=call(values);costs.append(dict(repeat=repeat,shape=label,**cost));del result
    median={label:statistics.median(v['wall_seconds'] for v in costs if v['shape']==label) for label in ['partial','full2048']}
    record.update(costs=costs,median_seconds=median,partial_to_full_cost_ratio=median['partial']/median['full2048'])
    state['checks'].append(record);save();print(json.dumps({k:v for k,v in record.items() if k!='costs'}),flush=True)
    del args
state.update(phase='completed',completed_unix=time.time());save()
