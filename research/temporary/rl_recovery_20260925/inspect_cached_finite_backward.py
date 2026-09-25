"""Existing finite FLA coincident-endpoint check on recorded cached Qwen data.

Uses native saved stage operands, the original finite callback/profile, and the
pinned FLA FP32 recurrence/assertions. This covers local backward consistency;
it does not declare a nonzero-intervention attribution vector exact.
"""
import importlib
import argparse
import json
import os
from pathlib import Path
import sys
import torch
import fla.utils
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from finite_fla_gpu import make_compiled_finite_pullback
from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders
from verify_official_kernel_tolerances import load

parser=argparse.ArgumentParser()
parser.add_argument('--compute-dtype',choices=('bfloat16','float16'),default='bfloat16')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
r=Path(os.environ['DT_RUNTIME_ROOT'])/'receipts'
saved=torch.load(r/'rollout-major-cost/native-prefix-gdn-operands.pt',map_location='cpu',weights_only=True,mmap=True)
part=saved['cached_gdn'];e=part['endpoints']
owner=load('pinned_fla_test',r/'training-setup/official-kernel-tests/test_gated_delta_v041.py')
assert not fla.utils.FLA_CI_ENV
torch.set_num_threads(8)
torch.manual_seed(42)
inputs=[e[n][1::2].cuda().detach().requires_grad_() for n in ('q','k','v','beta','raw_g')]
q,k,v,beta,g=inputs
initial=e['h'][1::2,0].cuda().float()
stage=importlib.import_module('fla.ops.gated_delta_rule.chunk').chunk_gated_delta_rule_fwd
endpoints={}
def capture(frame,event,value):
    if frame.f_code is stage.__code__ and event=='return' and value is not None:
        for n in ('q','k','v','g','beta','A','w','v_new','h'):
            endpoints[n]=frame.f_locals[n].detach().repeat_interleave(2,0)
assert sys.getprofile() is None
sys.setprofile(capture)
try:
    dtype=getattr(torch,args.compute_dtype)
    out,_=chunk_gated_delta_rule(q=q.to(dtype),k=k.to(dtype),v=v.to(dtype),beta=beta.to(dtype),g=g,scale=part['scale'],
        initial_state=initial,use_qk_l2norm_in_kernel=False)
finally:sys.setprofile(None)
endpoints['raw_g']=g.detach().repeat_interleave(2,0)
upstream=torch.randn_like(v).to(dtype)
native=torch.autograd.grad(out,inputs,upstream)
ref,_=owner.recurrent_gated_delta_rule_ref(q=q,k=k,v=v,beta=beta,g=g,scale=part['scale'],initial_state=initial)
reference=torch.autograd.grad(ref,inputs,upstream.float())
finite=average_memory_endpoint_orders(make_compiled_finite_pullback(dynamic_shapes=True))
with torch.no_grad():
    pieces=[finite({n:x[:,:,h:h+8].contiguous() for n,x in endpoints.items()},
                   upstream[:,:,h:h+8].contiguous(),part['scale']) for h in range(0,32,8)]
    coefficients={n:torch.cat([p[n] for p in pieces],dim=2) for n in pieces[0]}
result=dict(scope=__doc__,compute_dtype=args.compute_dtype,shape=list(q.shape),checks=[])
for variant,actual in [('native',dict(zip(('q','k','v','beta','g'),native))),('finite_head8',coefficients)]:
    for name,expected in zip(('q','k','v','beta','g'),reference):
        threshold=.02 if name in ('beta','g') else .008
        check=dict(variant=variant,name=name,threshold=threshold,
            rms_ratio=float(fla.utils.get_err_ratio(expected,actual[name])),
            max_abs=float(fla.utils.get_abs_err(expected,actual[name])))
        try:
            fla.utils.assert_close(variant+' d'+name,expected,actual[name],threshold);check['status']='passed'
        except AssertionError as exc:check.update(status='failed',error=str(exc))
        result['checks'].append(check);print(json.dumps(check),flush=True)
(args.output or r/'rollout-major-cost/native-cached-finite-backward.json').write_text(json.dumps(result,indent=2)+'\n')
