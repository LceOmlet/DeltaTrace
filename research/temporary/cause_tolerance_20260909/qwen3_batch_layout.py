"""Fuse only original attribution expression boundaries using installed Inductor.

No GEMM/model/FlashAttention replacement. FP32 arithmetic and original finite
rules retained; default compiler fusion rounding is explicitly accepted.
"""
import torch
from compiled_native_probability_audit import probability_with_audit
from signed_secant_rules import rmsnorm_secant_pullback


def midpoint_rule(a,b):
    return (a+b)*0.5


def scaled_probability_rule(qk,lse,scaling):
    return probability_with_audit(qk*scaling,lse)


def rotation_transpose(m,cos,sin):
    value=m*sin
    first,second=value.chunk(2,dim=-1)
    return m*cos-torch.cat((-second,first),dim=-1)


def attention_layout_rule(q_product,k_product,v_product,cos,sin,groups,scaling,k_heads,v_heads,n,h):
    mq=q_product*scaling
    mkr=k_product*scaling
    mk=mkr.reshape(mkr.shape[0],k_heads,groups,n,h).sum(2)
    mv=v_product.reshape(v_product.shape[0],v_heads,groups,n,h).sum(2)
    cos=cos.float().unsqueeze(1)
    sin=sin.float().unsqueeze(1)
    mqn=rotation_transpose(mq,cos,sin).transpose(1,2)
    mkn=rotation_transpose(mk,cos,sin).transpose(1,2)
    return mqn,mkn,mv


def norm_residual_two_rule(x0,x1,weight,first,second,residual,eps):
    upstream=first+second
    normalized=rmsnorm_secant_pullback(x0,x1,weight,upstream,eps)
    return residual+normalized


def norm_residual_three_rule(x0,x1,weight,first,second,third,residual,eps):
    upstream=first+second+third
    normalized=rmsnorm_secant_pullback(x0,x1,weight,upstream,eps)
    return residual+normalized


_midpoint=torch.compile(midpoint_rule,fullgraph=True,dynamic=True,backend='inductor')
_probability=torch.compile(scaled_probability_rule,fullgraph=True,dynamic=True,backend='inductor')
_layout=torch.compile(attention_layout_rule,fullgraph=True,dynamic=True,backend='inductor')
_norm_two=torch.compile(norm_residual_two_rule,fullgraph=True,dynamic=True,backend='inductor')
_norm_three=torch.compile(norm_residual_three_rule,fullgraph=True,dynamic=True,backend='inductor')


def midpoint(a,b):
    with torch.profiler.record_function('ATTR_COMPILED_MIDPOINT'):
        return _midpoint(a,b)


def scaled_probability(qk,lse,scaling):
    with torch.profiler.record_function('ATTR_COMPILED_SCALED_PROBABILITY'):
        return _probability(qk,lse,scaling)


def attention_layout(q_product,k_product,v_product,cos,sin,groups,scaling,k_heads,v_heads,n,h):
    with torch.profiler.record_function('ATTR_COMPILED_ATTENTION_LAYOUT'):
        return _layout(q_product,k_product,v_product,cos,sin,groups,scaling,k_heads,v_heads,n,h)


def norm_residual_two(x0,x1,weight,first,second,residual,eps):
    with torch.profiler.record_function('ATTR_COMPILED_NORM_RESIDUAL_TWO'):
        return _norm_two(x0,x1,weight,first,second,residual,eps)


def norm_residual_three(x0,x1,weight,first,second,third,residual,eps):
    with torch.profiler.record_function('ATTR_COMPILED_NORM_RESIDUAL_THREE'):
        return _norm_three(x0,x1,weight,first,second,third,residual,eps)
