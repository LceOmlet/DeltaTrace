"""Original FLA assertions on saved Qwen operands; no runtime or tolerance change."""
import json
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import fla.utils
from verify_official_kernel_tolerances import load

r=Path(os.environ['DT_RUNTIME_ROOT'])/'receipts'
out=r/'rollout-major-cost/recovery-fla-native-backward.json'
owner=load('pinned_fla_test',r/'training-setup/official-kernel-tests/test_gated_delta_v041.py')
assert not fla.utils.FLA_CI_ENV
torch.set_num_threads(8)
result={'checks':[], 'scope':__doc__}
try:
    owner.test_chunk(B=4,T=1024,H=4,D=128,scale=.1,gate_logit_normalizer=1.,mask_p=0.,
                     use_qk_l2norm_in_kernel=True,dtype=torch.float16)
    result['original_fp16_case']='passed'
except AssertionError as e:result['original_fp16_case']=str(e)
saved=torch.load(r/'rollout-major-cost/author-native-operators.pt',map_location='cpu',weights_only=True)
values,endpoints=saved['values'],saved['endpoints']
inputs=[values['raw_q'][:2],values['raw_k'][:2],endpoints['v'][:2],endpoints['beta'][:2],endpoints['raw_g'][:2]]
result['shapes']=[list(x.shape) for x in inputs]
torch.manual_seed(42)
up=torch.randn_like(inputs[2],device='cuda')
answers=[]
for implementation in ['native','reference']:
    q,k,v,beta,g=[x.cuda().detach().requires_grad_() for x in inputs]
    if implementation=='native':
        y,_=owner.chunk_gated_delta_rule(q=q,k=k,v=v,beta=beta,g=g,scale=saved['scale'],use_qk_l2norm_in_kernel=True)
    else:
        y,_=owner.recurrent_gated_delta_rule_ref(q=F.normalize(q,p=2,dim=-1),k=F.normalize(k,p=2,dim=-1),v=v,beta=beta,g=g,scale=saved['scale'])
    (y*up).sum().backward()
    answers.append([x.detach().cpu() for x in [y,q.grad,k.grad,v.grad,beta.grad,g.grad]])
    del y,q,k,v,beta,g
for name,threshold,actual,reference in zip(['o','dq','dk','dv','db','dg'],[.005,.008,.008,.008,.02,.02],answers[0],answers[1]):
    check=dict(name=name,threshold=threshold,rms_ratio=float(fla.utils.get_err_ratio(reference,actual)),max_abs=float(fla.utils.get_abs_err(reference,actual)))
    try:
        fla.utils.assert_close(name,reference,actual,threshold);check['status']='passed'
    except AssertionError as e:check.update(status='failed',error=str(e))
    result['checks'].append(check)
    out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
