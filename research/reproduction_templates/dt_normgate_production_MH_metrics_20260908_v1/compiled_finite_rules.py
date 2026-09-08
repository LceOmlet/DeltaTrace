"""Existing Inductor on explicit finite attribution rules, never on the model.

The original mathematical expressions and FP32 analysis inputs are retained.
The softmax validation predicates are reduced by the compiled graph and checked
by the wrapper; no eager fallback is allowed through fullgraph=True.
"""
import torch
from signed_secant_rules import rmsnorm_secant_pullback as original_rmsnorm


def logarithmic_mean_with_checks(z0,z1):
    lp0=z0.log_softmax(-1)
    lp1=z1.log_softmax(-1)
    valid=torch.isfinite(lp0)&torch.isfinite(lp1)
    same_mask=(torch.isfinite(lp0)==torch.isfinite(lp1)).all()
    distance=torch.where(valid,(lp1-lp0).abs(),torch.zeros_like(lp0))
    denominator=torch.where(distance>0,distance,torch.ones_like(distance))
    ratio=-torch.expm1(-distance)/denominator
    ratio=torch.where(distance==0,torch.ones_like(ratio),ratio)
    mean=torch.where(valid,torch.maximum(lp0,lp1).exp()*ratio,torch.zeros_like(lp0))
    checks=same_mask & torch.isfinite(mean).all() & (mean>=0).all() & (mean.sum(-1)>0).all()
    return mean,checks


def softmax_pullback_with_checks(z0,z1,upstream):
    mean,checks=logarithmic_mean_with_checks(z0,z1)
    center=(mean*upstream).sum(-1,keepdim=True)/mean.sum(-1,keepdim=True)
    return mean*(upstream-center),checks


compiled_softmax=torch.compile(softmax_pullback_with_checks,fullgraph=True,dynamic=True,backend='inductor')
compiled_rmsnorm=torch.compile(original_rmsnorm,fullgraph=True,dynamic=True,backend='inductor')


def softmax_secant_pullback(z0,z1,upstream):
    result,checks=compiled_softmax(z0,z1,upstream)
    assert bool(checks),'Finite softmax validation failed'
    return result


def rmsnorm_secant_pullback(x0,x1,weight,upstream,eps):
    return compiled_rmsnorm(x0,x1,weight,upstream,eps)
