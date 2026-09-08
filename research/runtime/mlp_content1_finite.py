"""Optional DT MLP finite allocation; never changes the model or FlashTrace.

For a=SiLU(g), allocate Delta(u*a)=a1*Delta(u)+u0*Delta(a).
The symmetric DT rule remains the default in FiniteBoundaryOps. This explicit
candidate replaces only its attribution MLP callback and adds no model calls.
"""
import torch
from qwen35_decoder_finite import FiniteBoundaryOps, _linear_transpose, _secant


def swiglu_content1_finite_rule(gate0,gate1,up0,up1,silu0,silu1,m_product):
    gate0=gate0.float();gate1=gate1.float();up0=up0.float()
    sigmoid=gate0.sigmoid()
    derivative=sigmoid*(1+gate0*(1-sigmoid))
    multiplier=_secant(gate0,gate1,silu0.float(),silu1.float(),derivative)
    return m_product*silu1.float(),m_product*up0*multiplier


def mlp_content1_input_rule(g0,g1,u0,u1,s0,s1,upstream,down_weight,up_weight,gate_weight):
    product=_linear_transpose(upstream,down_weight)
    mu,mg=swiglu_content1_finite_rule(g0,g1,u0,u1,s0,s1,product)
    return _linear_transpose(mu,up_weight)+_linear_transpose(mg,gate_weight)


class Content1MLPBoundaryOps(FiniteBoundaryOps):
    def __init__(self,compiled=True):
        super().__init__(compiled)
        self.mlp=(torch.compile(mlp_content1_input_rule,fullgraph=True,dynamic=False,
            options={'triton.cudagraphs':False,'max_autotune':False}) if compiled else mlp_content1_input_rule)
