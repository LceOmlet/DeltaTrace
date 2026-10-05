"""Candidate local finite-difference rules, independent of FlashTrace scores.

These are attribution operators, not the model's native Jacobians and not yet
an end-to-end Qwen attribution method. No rule clips negative contributions.
Input tensors should already be in the chosen analysis precision.
"""

def softmax_logarithmic_mean(z0,z1):
    import torch
    lp0=z0.log_softmax(-1)
    lp1=z1.log_softmax(-1)
    valid=torch.isfinite(lp0)&torch.isfinite(lp1)
    assert torch.equal(torch.isfinite(lp0),torch.isfinite(lp1))
    distance=torch.where(valid,(lp1-lp0).abs(),torch.zeros_like(lp0))
    denominator=torch.where(distance>0,distance,torch.ones_like(distance))
    ratio=-torch.expm1(-distance)/denominator
    ratio=torch.where(distance==0,torch.ones_like(ratio),ratio)
    mean=torch.where(valid,torch.maximum(lp0,lp1).exp()*ratio,torch.zeros_like(lp0))
    assert torch.isfinite(mean).all() and (mean>=0).all() and (mean.sum(-1)>0).all()
    return mean


def softmax_secant_pullback(z0,z1,upstream):
    """Symmetric PSD finite secant: diag(L)-L L^T/sum(L), applied without a matrix."""
    mean=softmax_logarithmic_mean(z0,z1)
    center=(mean*upstream).sum(-1,keepdim=True)/mean.sum(-1,keepdim=True)
    return mean*(upstream-center)


def logprob_secant_seed(z0,z1,target):
    """Exact-real-arithmetic finite seed for log softmax(z)[target]."""
    mean=softmax_logarithmic_mean(z0,z1)
    seed=-mean/mean.sum(-1,keepdim=True)
    seed=seed.clone()
    seed.scatter_add_(-1,target.unsqueeze(-1),seed.new_ones((*target.shape,1)))
    return seed


def matmul_secant_pullback(a0,a1,b0,b1,upstream):
    """Transpose action of Δ(AB)=ΔA·mean(B)+mean(A)·ΔB."""
    abar=(a0+a1)*.5
    bbar=(b0+b1)*.5
    return upstream@bbar.transpose(-1,-2),abar.transpose(-1,-2)@upstream


def rmsnorm_secant_pullback(x0,x1,weight,upstream,eps):
    """Analytical finite secant for weight*x/sqrt(mean(x²)+eps)."""
    r0=(x0.square().mean(-1,keepdim=True)+eps).sqrt()
    r1=(x1.square().mean(-1,keepdim=True)+eps).sqrt()
    inverse_mean=(1/r0+1/r1)*.5
    inverse_secant=-1/(r0*r1*(r0+r1))
    xbar=(x0+x1)*.5
    weighted=upstream*weight
    coefficient=2*inverse_secant/x0.shape[-1]
    return inverse_mean*weighted+xbar*coefficient*(xbar*weighted).sum(-1,keepdim=True)


def nonlinear_reveal_cancel(function,base,positive_incoming,negative_incoming):
    """Two-group conditional allocation; incoming signs need not be output signs.

    Local evaluations are values of the attribution rule, not extra native
    whole-model counterfactual forwards. This is not global input Shapley.
    """
    assert (positive_incoming>=0).all() and (negative_incoming<=0).all()
    base_value=function(base)
    plus_value=function(base+positive_incoming)
    minus_value=function(base+negative_incoming)
    joint_value=function(base+positive_incoming+negative_incoming)
    positive_credit=.5*((plus_value-base_value)+(joint_value-minus_value))
    negative_credit=.5*((minus_value-base_value)+(joint_value-plus_value))
    return positive_credit,negative_credit
