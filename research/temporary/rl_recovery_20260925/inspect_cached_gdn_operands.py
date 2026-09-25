"""Apply the pinned FLA forward assertion to actual full/cached GDN operands.

This isolates original forward/cache arithmetic; no finite DT or training code
is replaced. Full-reference comparison includes the native prefix-state error;
local comparison conditions on the native cached initial state explicitly.
"""
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import fla.utils
from verify_official_kernel_tolerances import load

r = Path(os.environ['DT_RUNTIME_ROOT'])/'receipts'
saved = torch.load(r/'rollout-major-cost/native-prefix-gdn-operands.pt',map_location='cpu',weights_only=True)
cut = saved['cut']
owner = load('pinned_fla_test',r/'training-setup/official-kernel-tests/test_gated_delta_v041.py')
assert not fla.utils.FLA_CI_ENV
torch.set_num_threads(8)
result = dict(scope=__doc__, cut=cut,operand_differences={},checks=[])
full,cached = saved['full_gdn'],saved['cached_gdn']
for container,names in [('values',['raw_q','raw_k','a','b','conv_output']),
                        ('endpoints',['q','k','v','beta','raw_g','o','h'])]:
    for name in names:
        a,b = full[container][name],cached[container][name]
        dim,start = (2,cut) if name=='conv_output' else (1,cut//64 if name=='h' else cut)
        a = a.narrow(dim,start,a.shape[dim]-start)
        result['operand_differences'][name] = dict(equal=torch.equal(a,b),
            max_abs=float((a.float()-b.float()).abs().max()),dtype=str(a.dtype))

@torch.no_grad()
def reference(part,initial=None):
    v,e=part['values'],part['endpoints']
    return owner.recurrent_gated_delta_rule_ref(q=F.normalize(v['raw_q'][:2].cuda(),p=2,dim=-1),
        k=F.normalize(v['raw_k'][:2].cuda(),p=2,dim=-1),v=e['v'][:2].cuda(),
        beta=e['beta'][:2].cuda(),g=e['raw_g'][:2].cuda(),scale=part['scale'],initial_state=initial)[0]

full_ref = reference(full)
local_ref = reference(cached,cached['endpoints']['h'][:2,0].cuda().float())
for name,ref,actual in [
    ('full_native',full_ref,full['endpoints']['o'][:2].cuda()),
    ('cached_vs_full_reference',full_ref[:,cut:],cached['endpoints']['o'][:2].cuda()),
    ('cached_given_native_initial_state',local_ref,cached['endpoints']['o'][:2].cuda())]:
    check=dict(name=name,threshold=.005,rms_ratio=float(fla.utils.get_err_ratio(ref,actual)),
               max_abs=float(fla.utils.get_abs_err(ref,actual)))
    try:
        fla.utils.assert_close(name,ref,actual,.005);check['status']='passed'
    except AssertionError as exc:check.update(status='failed',error=str(exc))
    result['checks'].append(check)
    print(json.dumps(check),flush=True)
(r/'rollout-major-cost/native-cached-gdn-reference.json').write_text(json.dumps(result,indent=2)+'\n')
