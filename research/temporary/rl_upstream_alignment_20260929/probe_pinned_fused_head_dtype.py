"""Same-operand pinned VERL head check, including the actual BF16 autocast path."""
import argparse
import json
from pathlib import Path
import time

import torch
import verl.utils.torch_functional as VF
from verl.utils.experimental.torch_functional import FusedLinearForPPO

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--temperature',type=float,default=1.)
args=p.parse_args()
torch.manual_seed(123)
# Match the production owner wrapper: the chunk kernels are compiled, while
# the existing chunk loop stays eager rather than being unrolled by Dynamo.
fused=FusedLinearForPPO()
entropy=torch.compile(VF.entropy_from_logits,dynamic=True)
result={'owner':VF.__file__,'batch':4,'tokens':64,'hidden':4096,'vocab':248320,
        'temperature':args.temperature,'cases':[]}
for dtype in [torch.float32,torch.bfloat16]:
    x=torch.empty(4,64,4096,device='cuda',dtype=dtype).uniform_(-.5,.5)
    weight=torch.empty(248320,4096,device='cuda',dtype=dtype).uniform_(-.5,.5)
    ids=torch.randint(248320,(4,64),device='cuda')
    upstream=(torch.randn(4,64,device='cuda'),torch.randn(4,64,device='cuda'))
    outputs=[]
    for name in ['dense','fused']:
        hidden=x.detach().clone().requires_grad_()
        start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
            if name=='dense':
                logits=(hidden @ weight.t())/args.temperature
                lp=VF.logprobs_from_logits(logits,ids,inplace_backward=False)
                en=entropy(logits)
            else:
                lp,en=fused(hidden,weight,ids,args.temperature)
        grad,=torch.autograd.grad((lp,en),hidden,upstream)
        torch.cuda.synchronize()
        outputs.append((lp.detach(),en.detach(),grad.detach()))
        print(json.dumps(dict(dtype=str(dtype),mode=name,seconds=time.perf_counter()-start)),flush=True)
    record=dict(dtype=str(dtype),output_dtypes=[[str(y.dtype) for y in a] for a in outputs],
                max_abs=[(a.float()-b.float()).abs().max().item() for a,b in zip(*outputs)],assertions=[])
    # The pinned official head test's unchanged forward and backward thresholds.
    for i,(a,b) in enumerate(zip(*outputs)):
        try:
            torch.testing.assert_close(a,b,atol=1e-4 if i<2 else 1e-2,rtol=1e-4)
            record['assertions'].append({'passed':True})
        except AssertionError as exc:
            record['assertions'].append({'passed':False,'failure':str(exc)})
    result['cases'].append(record)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    del x,weight,outputs,hidden,grad,lp,en
