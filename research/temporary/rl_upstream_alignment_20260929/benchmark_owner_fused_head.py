"""Warm timing of corresponding pinned dense/fused heads, on identical inputs."""
import argparse,json,time
from pathlib import Path
import torch
import verl.utils.torch_functional as VF
from verl.utils.experimental.torch_functional import FusedLinearForPPO

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--temperature',type=float,default=1.);args=p.parse_args()
torch.manual_seed(9)
x=torch.empty(4,512,4096,device='cuda',dtype=torch.bfloat16).uniform_(-.5,.5)
w=torch.empty(248320,4096,device='cuda',dtype=torch.bfloat16).uniform_(-.5,.5)
ids=torch.randint(248320,(4,512),device='cuda')
up=(torch.randn(4,512,device='cuda'),torch.randn(4,512,device='cuda'))
entropy=torch.compile(VF.entropy_from_logits,dynamic=True)
fused=FusedLinearForPPO()
report=dict(shape=list(x.shape),vocab=248320,temperature=args.temperature,cases=[])
for mode in ['dense','fused']:
    values=[]
    for repeat in range(4):
        h=x.detach().requires_grad_()
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if mode=='dense':
                logits=(h@w.t())/args.temperature
                lp=VF.logprobs_from_logits(logits,ids,inplace_backward=False)
                en=entropy(logits)
            else:
                lp,en=fused(h,w,ids,args.temperature)
        torch.cuda.synchronize();f=time.perf_counter()
        grad,=torch.autograd.grad((lp,en),h,up)
        torch.cuda.synchronize();end=time.perf_counter()
        values.append(dict(forward=f-t,backward=end-f,total=end-t))
        del lp,en,grad,h
        if mode=='dense': del logits
    report['cases'].append(dict(mode=mode,cold=values[0],warm=values[1:]))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['cases'][-1]),flush=True)
