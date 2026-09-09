"""Experimental finite normalization coefficient, never a model replacement.

For L2 normalization without epsilon, project onto the plane perpendicular to
n0+n1 and scale by 2/(r0+r1). This preserves the endpoint secant and annihilates
the repeated-variable upstream of n.T@n. For RMSNorm epsilon, append a constant
sqrt(d*epsilon) coordinate algebraically; no tensor dimension is appended.
This fixes this normalization identity, not arbitrary graph invariance.
"""
import torch


def rmsnorm_geometry_pullback(x0,x1,weight,upstream,eps):
    dimension=x0.shape[-1]
    radius0=(x0.square().sum(-1,keepdim=True)+dimension*eps).sqrt()
    radius1=(x1.square().sum(-1,keepdim=True)+dimension*eps).sqrt()
    normal=x0/radius0+x1/radius1
    # Extra constant coordinate contributes only to the squared normal norm.
    normal_square=normal.square().sum(-1,keepdim=True)+dimension*eps*(1/radius0+1/radius1).square()
    weighted=upstream*weight*dimension**.5
    dot=(normal*weighted).sum(-1,keepdim=True)
    denominator=torch.where(normal_square>0,normal_square,torch.ones_like(normal_square))
    return 2/(radius0+radius1)*(weighted-normal*(dot/denominator))


def l2_geometry_pullback(x0,x1,upstream):
    dimension=x0.shape[-1]
    return rmsnorm_geometry_pullback(x0.float(),x1.float(),dimension**-.5,upstream,1e-6/dimension)
