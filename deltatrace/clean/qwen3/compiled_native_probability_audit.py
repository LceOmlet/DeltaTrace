"""Compile explicit attribution operands and their complete original audits.

The raw scores are computed outside this helper from the actual native Q/K.
The LSE and actual output are captured from the unchanged vendor FlashAttention.
These probabilities are attribution operands only: never returned to the model.
The compiler is the installed Inductor; no custom attention or backward kernel.
"""
import torch


def probability_with_audit(raw,lse):
    n=raw.shape[-1]
    invalid=torch.arange(n,device=raw.device)[None,:]>torch.arange(n,device=raw.device)[:,None]
    z=raw.masked_fill(invalid,-torch.inf)
    prob=(z-lse.unsqueeze(-1)).exp()
    row_sum=prob.sum(-1)
    # Equivalent to masked_select(invalid).eq(0).all(), without constructing
    # the selected array or data-dependent nonzero indices.
    checks=torch.isfinite(prob).all() & (prob>=0).all() & (row_sum>0).all() & ((~invalid)|(prob==0)).all()
    reference=z.softmax(-1)
    relative=(prob-reference).norm()/reference.norm()
    row_error=(row_sum-1).abs().max()
    return z,prob,torch.stack((checks.to(dtype=prob.dtype),relative,row_error))


def pv_with_audit(pv,actual):
    return torch.stack(((pv-actual).norm()/actual.norm(),(pv-actual).abs().max()))


compiled_probability=torch.compile(probability_with_audit,fullgraph=True,dynamic=True,backend='inductor')
compiled_pv_audit=torch.compile(pv_with_audit,fullgraph=True,dynamic=True,backend='inductor')
