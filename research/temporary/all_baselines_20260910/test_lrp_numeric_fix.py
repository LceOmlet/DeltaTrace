"""Exercise FP16 zeros, ordinary finite ratios, and restoration on exceptions."""
import os
import sys
os.environ.setdefault('MACA_PATH','/opt/maca')
sys.path.insert(0,sys.argv[1])
import torch
import lrp_rules
from lrp_numeric_fix import zero_ratio_guard

class Capture:
    def save_for_backward(self,*values):self.values=values

original=lrp_rules.IdentityRuleImplicitFn.forward
values=torch.tensor([0.,-0.,1e-6,-1e-6,.5,-.5,8.,-8.],dtype=torch.float16,requires_grad=True)
native=Capture();out=original(native,torch.nn.functional.silu,values)
assert int((~torch.isfinite(native.values[0])).sum())==2
with zero_ratio_guard() as receipt:
    fixed=Capture();other=lrp_rules.IdentityRuleImplicitFn.forward(fixed,torch.nn.functional.silu,values)
    assert torch.equal(out,other)
    mask=torch.isfinite(native.values[0]);assert torch.equal(native.values[0][mask],fixed.values[0][mask])
    assert torch.isfinite(fixed.values[0]).all() and not fixed.values[0][~mask].any()
    result=lrp_rules.identity_rule_implicit(torch.nn.functional.silu,values)
    result.sum().backward();assert torch.isfinite(values.grad).all()
assert lrp_rules.IdentityRuleImplicitFn.forward is original and receipt['repaired_zero_ratios']==4
try:
    with zero_ratio_guard():raise RuntimeError('test restoration')
except RuntimeError:pass
assert lrp_rules.IdentityRuleImplicitFn.forward is original
print('PASS: unchanged finite ratios and outputs; repaired FP16 zero ratios; function restored')
