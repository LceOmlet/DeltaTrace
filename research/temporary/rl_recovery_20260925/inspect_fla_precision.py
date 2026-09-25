"""Compare supported native FLA input dtypes on the same saved operands.

No model or DT code changes. Record cast error rather than requiring a change
of dtype to be lossless. Gradients return to the original BF16 inputs.
The existing FLA reference and thresholds are kept unchanged.
"""
import json
import os
from pathlib import Path
import time
import torch
import fla.utils
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from verify_official_kernel_tolerances import load

r=Path(os.environ['DT_RUNTIME_ROOT'])/'receipts'
saved=torch.load(r/'rollout-major-cost/native-prefix-gdn-operands.pt',map_location='cpu',weights_only=True,mmap=True)
p=saved['cached_gdn'];e=p['endpoints']
owner=load('pinned_fla_test',r/'training-setup/official-kernel-tests/test_gated_delta_v041.py')
torch.set_num_threads(8);torch.manual_seed(42)
assert not fla.utils.FLA_CI_ENV
initial=e['h'][1::2,0].cuda().float()
original=[e[n][1::2].cuda().detach() for n in ('q','k','v','beta','raw_g')]
upstream=torch.randn_like(original[2])
inputs=[x.clone().requires_grad_() for x in original]
ref,_=owner.recurrent_gated_delta_rule_ref(q=inputs[0],k=inputs[1],v=inputs[2],beta=inputs[3],g=inputs[4],scale=p['scale'],initial_state=initial)
reference=torch.autograd.grad(ref,inputs,upstream.float())
result=dict(scope=__doc__,shape=list(original[0].shape),cases=[])
for dtype in (torch.float16,):
    case=dict(dtype=str(dtype),seconds=[],checks=[])
    try:
        for repeat in range(2):
            inputs=[x.clone().requires_grad_() for x in original]
            q,k,v,beta,g=inputs
            converted=[x.to(dtype) for x in (q,k,v,beta)]
            case['input_cast_max_abs']=[float((x.float()-y.float()).abs().max()) for x,y in zip((q,k,v,beta),converted)]
            torch.cuda.synchronize();started=time.perf_counter()
            out,_=chunk_gated_delta_rule(q=converted[0],k=converted[1],v=converted[2],beta=converted[3],g=g,
                scale=p['scale'],initial_state=initial,use_qk_l2norm_in_kernel=False)
            grads=torch.autograd.grad(out,inputs,upstream.to(dtype))
            torch.cuda.synchronize();case['seconds'].append(time.perf_counter()-started)
        for name,expected,actual in zip(('q','k','v','beta','g'),reference,grads):
            threshold=.02 if name in ('beta','g') else .008
            check=dict(name=name,threshold=threshold,rms_ratio=float(fla.utils.get_err_ratio(expected,actual)))
            try:
                fla.utils.assert_close(str(dtype)+' d'+name,expected,actual,threshold);check['status']='passed'
            except AssertionError as exc:check.update(status='failed',error=str(exc))
            case['checks'].append(check)
    except Exception as exc:case['error']=repr(exc)
    result['cases'].append(case)
    (r/'rollout-major-cost/native-fla-fp16.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(case),flush=True)
