"""Finite normalization operator: native-definition J1 on off-chord directions.

This attribution rule is J1 + (delta_n-J1*delta_x) delta_x^T / ||delta_x||^2.
It is the unique correction supported only on the endpoint chord. It preserves
the real normalization endpoint relation, not any unobserved partial relation.
"""
def input_supported_l2_pullback(x0,x1,upstream,eps=1e-6):
    import torch
    r0=(x0.square().sum(-1,keepdim=True)+eps).sqrt()
    r1=(x1.square().sum(-1,keepdim=True)+eps).sqrt()
    native_definition_j1=upstream/r1-x1*(x1*upstream).sum(-1,keepdim=True)/r1.pow(3)
    delta=x1-x0;length2=delta.square().sum(-1,keepdim=True)
    defect=(upstream*(x1/r1-x0/r0)).sum(-1,keepdim=True)-(native_definition_j1*delta).sum(-1,keepdim=True)
    correction=defect/torch.where(length2>0,length2,torch.ones_like(length2))
    return native_definition_j1+delta*correction
