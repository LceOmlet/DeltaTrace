"""Bounded transport-only comparison for the captured gate's native layout.

Uses public Torch copies, no model/DT/update. Shapes reproduce paired B8 gate
storage and original2048-token slices; input values are explicit transport
fixtures, not reconstructed training activations or task quality examples.
"""
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

AUDIT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-dt-cost-breakdown-20261016-v1')
text=subprocess.check_output(['mx-smi'],text=True)
assert not any(re.match(r'\|\s*6\s+\d+\s+\S+',line) for line in text.splitlines()),'GPU6 occupied; no process was stopped'
import torch
assert torch.cuda.device_count()==1
torch.cuda.set_device(0)
state=dict(started_unix=time.time(),scope=__doc__,physical_GPU=6,checks=[],cases=[],
    tolerance=dict(atol=0,rtol=0,reason='Identity storage transport of the same BF16 values'),
    numerical_DT_claim=False,model_calls=0,optimizer_calls=0)
path=AUDIT/'strided-gate-transport.json'
def save():path.write_text(json.dumps(state,indent=2)+'\n')
def transfer(value,direct):
    if direct:return torch.empty(value.shape,device='cuda',dtype=value.dtype).copy_(value,non_blocking=True)
    return value.to('cuda',non_blocking=True).contiguous()
for length in [6906,10065,16250,32768]:
    state['phase']=f'paired_B8_T{length}';save()
    source=torch.empty((8,length,12288),dtype=torch.bfloat16,pin_memory=True)
    for row in range(8):source[row].fill_(row+.125)
    groups=[source[0::2].split(2048,dim=1),source[1::2].split(2048,dim=1)]
    for index in [0,len(groups[0])-1]:
        value=groups[0][index];left=transfer(value,False);right=transfer(value,True)
        equal=torch.equal(left,right);assert equal
        state['checks'].append(dict(length=length,chunk=index,shape=list(value.shape),stride=list(value.stride()),
            pinned=value.is_pinned(),outputs_bitwise_equal=equal,contiguous_left=left.is_contiguous(),contiguous_right=right.is_contiguous()))
        del left,right
    def run(direct):
        torch.cuda.synchronize();tick=time.perf_counter();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        for pair in zip(*groups):
            values=[transfer(v,direct) for v in pair]
            del values
        end.record();torch.cuda.synchronize()
        return dict(wall_seconds=time.perf_counter()-tick,stream_elapsed_seconds=start.elapsed_time(end)/1000)
    costs=[]
    for repeat in range(4):
        order=[False,True] if repeat%2==0 else [True,False]
        for direct in order:costs.append(dict(repeat=repeat,mode='direct_destination_copy' if direct else 'original_to_contiguous',**run(direct)))
    median={label:statistics.median(v['wall_seconds'] for v in costs if v['mode']==label) for label in ['original_to_contiguous','direct_destination_copy']}
    row=dict(length=length,paired_gate_bytes=source.numel()*source.element_size(),chunks=len(groups[0]),costs=costs,median_wall_seconds=median,
        component_ratio=median['original_to_contiguous']/median['direct_destination_copy'],physical_free_bytes=torch.cuda.mem_get_info()[0])
    state['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='costs'}),flush=True)
    del groups,source
state.update(phase='completed',completed_unix=time.time());save()
