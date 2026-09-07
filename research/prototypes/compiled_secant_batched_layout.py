"""Existing finite GQA/RoPE layout, generalized leading sample dimension.
Official torch.compile only; not a model or FA kernel.
"""
import torch
from compiled_secant_boundaries import rotation_transpose
def attention_layout_rule(q_product,k_product,v_product,cos,sin,groups,scaling,k_heads,v_heads,n,h):
    mq=q_product*scaling
    mkr=k_product*scaling
    mk=mkr.reshape(q_product.shape[0],k_heads,groups,n,h).sum(2)
    mv=v_product.reshape(q_product.shape[0],v_heads,groups,n,h).sum(2)
    cos=cos.float().unsqueeze(1)
    sin=sin.float().unsqueeze(1)
    mqn=rotation_transpose(mq,cos,sin).transpose(1,2)
    mkn=rotation_transpose(mk,cos,sin).transpose(1,2)
    return mqn,mkn,mv
_layout=torch.compile(attention_layout_rule,fullgraph=True,dynamic=True,backend='inductor')
def attention_layout(*args):
    with torch.profiler.record_function('ATTR_COMPILED_ATTENTION_LAYOUT_BATCHED'):
        return _layout(*args)
