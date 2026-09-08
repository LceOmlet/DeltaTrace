"""Analytic path-averaged L2 Jacobian pullback, O(D) memory and arithmetic.

Attribution operator only. Does not run or replace native normalization.
Integral over x(t)=x0+t*(x1-x0) of J[x/sqrt(sum(x*x)+eps)].
"""
def integrated_l2_pullback(x0,x1,upstream,eps=1e-6):
    import torch
    delta=x1-x0
    length=delta.norm(dim=-1,keepdim=True)
    divisor=torch.where(length>0,length,torch.ones_like(length))
    direction=delta/divisor
    z0=(x0*direction).sum(-1,keepdim=True);z1=z0+length
    orthogonal=x0-z0*direction
    b2=orthogonal.square().sum(-1,keepdim=True)+eps
    r0=(z0.square()+b2).sqrt();r1=(z1.square()+b2).sqrt()
    crossing=(z0<0)&(z1>0)
    # Rationalize near an antipodal path; no clipping of coefficients or scores.
    denominator=torch.where(crossing,b2/(r1+z1)+b2/(r0-z0),r0+r1-length)
    a=torch.log1p(2*length/denominator)/divisor
    a=torch.where(length>0,a,1/r0)
    b_same=(r0+r1)/(r0*r1*(r0*r1+z0*z1+b2))
    b_cross=(z1/r1-z0/r0)/(divisor*b2)
    b=torch.where(crossing,b_cross,b_same)
    c=(z1+z0)/((r1+r0)*r0*r1)
    d=a-b2*b
    orthogonal_dot=(orthogonal*upstream).sum(-1,keepdim=True)
    direction_dot=(direction*upstream).sum(-1,keepdim=True)
    return a*upstream-b*orthogonal*orthogonal_dot-c*(orthogonal*direction_dot+direction*orthogonal_dot)-d*direction*direction_dot
