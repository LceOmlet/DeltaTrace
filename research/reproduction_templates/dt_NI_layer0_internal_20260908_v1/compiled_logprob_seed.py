"""Fuse the existing finite log-probability seed with installed Inductor.

Only explicit attribution mathematics is compiled. The model and its ordinary
forward/backward remain native. Output and checks retain FP32 calculation.
"""
import torch
from compiled_finite_rules import logarithmic_mean_with_checks


def seed_with_checks(z0,z1,target):
    mean,checks=logarithmic_mean_with_checks(z0,z1)
    seed=-mean/mean.sum(-1,keepdim=True)
    seed=seed.clone()
    seed.scatter_add_(-1,target.unsqueeze(-1),seed.new_ones((*target.shape,1)))
    return seed,checks


compiled_seed=torch.compile(seed_with_checks,fullgraph=True,dynamic=True,backend='inductor')


def logprob_secant_seed(z0,z1,target):
    seed,checks=compiled_seed(z0,z1,target)
    assert bool(checks),'Finite logprob seed validation failed'
    return seed
