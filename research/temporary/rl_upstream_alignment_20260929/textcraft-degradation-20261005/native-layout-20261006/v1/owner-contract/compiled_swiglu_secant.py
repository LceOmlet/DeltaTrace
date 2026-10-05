"""Fuse the original finite SwiGLU attribution formula with installed Inductor.

This is an attribution-only graph, never the model's SiLU/product or backward.
Actual native FP16 SiLU endpoints/product checks remain in the caller. FP32
arithmetic is retained, while default compiler fusion rounding is accepted.
"""
import torch


def swiglu_finite_rule(gate0,gate1,up0,up1,silu0,silu1,m_product):
    gate0=gate0.float();gate1=gate1.float();up0=up0.float();up1=up1.float()
    mg=m_product*(up0+up1)*0.5
    mu=m_product*(silu0+silu1)*0.5
    difference=gate1-gate0
    sigmoid=torch.sigmoid(gate0)
    derivative=sigmoid*(1+gate0*(1-sigmoid))
    multiplier=torch.where(difference!=0,(silu1-silu0)/torch.where(difference!=0,difference,torch.ones_like(difference)),derivative)
    m_gate=mg*multiplier
    return mu,m_gate


_compiled=torch.compile(swiglu_finite_rule,fullgraph=True,dynamic=True,backend='inductor')


def swiglu_secant_multipliers(gate0,gate1,up0,up1,silu0,silu1,m_product):
    with torch.profiler.record_function('ATTR_COMPILED_SWIGLU_SECANT'):
        return _compiled(gate0,gate1,up0,up1,silu0,silu1,m_product)
