"""Move existing FP16->FP32 casts into the unchanged compiled finite rules.

All input tensors are converted on every call. No static tensor cache, native
GEMM change, or new secant expression is introduced.
"""
import torch
from signed_secant_rules import rmsnorm_secant_pullback
from compiled_secant_boundaries import norm_residual_two_rule,norm_residual_three_rule,attention_layout_rule


def norm_rule(x0,x1,weight,upstream,eps):
    return rmsnorm_secant_pullback(x0.float(),x1.float(),weight.float(),upstream,eps)


def two_rule(x0,x1,weight,first,second,residual,eps):
    return norm_residual_two_rule(x0.float(),x1.float(),weight.float(),first,second,residual,eps)


def three_rule(x0,x1,weight,first,second,third,residual,eps):
    return norm_residual_three_rule(x0.float(),x1.float(),weight.float(),first,second,third,residual,eps)


def qk_rule(q0,q1,qweight,mq,qeps,k0,k1,kweight,mk,keps):
    return norm_rule(q0,q1,qweight,mq,qeps),norm_rule(k0,k1,kweight,mk,keps)


def layout_rule(q,k,v,cos,sin,groups,scaling,k_heads,v_heads,n,h):
    return attention_layout_rule(q.float(),k.float(),v.float(),cos,sin,groups,scaling,k_heads,v_heads,n,h)


def compile(fn):
    return torch.compile(fn,fullgraph=True,dynamic=True,
        options={'triton.cudagraphs':False,'max_autotune':False})


norm=compile(norm_rule)
norm_residual_two=compile(two_rule)
norm_residual_three=compile(three_rule)
qk_norms=compile(qk_rule)
attention_layout=compile(layout_rule)
