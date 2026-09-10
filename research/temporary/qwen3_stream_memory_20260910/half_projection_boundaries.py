"""Fuse only casts already required by the following native FP16-input GEMM."""
import torch
from compiled_swiglu_secant import swiglu_finite_rule
from compiled_logprob_seed import seed_with_checks


def swiglu_half_rule(*args):
    up,gate=swiglu_finite_rule(*args)
    return up.to(torch.float16),gate.to(torch.float16)


def seed_half_rule(*args):
    seed,checks=seed_with_checks(*args)
    return seed.to(torch.float16),checks


options={'triton.cudagraphs':False,'max_autotune':False,'emulate_precision_casts':True}
swiglu_half=torch.compile(swiglu_half_rule,fullgraph=True,dynamic=True,options=options)
seed_half=torch.compile(seed_half_rule,fullgraph=True,dynamic=True,options=options)


def native_precast_half_linear(multiplier,weight):
    assert multiplier.dtype==weight.dtype==torch.float16
    assert multiplier.shape[-1]==weight.shape[0] and weight.ndim==2
    with torch.profiler.record_function('ATTR_NATIVE_PRECAST_HALF_LINEAR'):
        flat=multiplier.reshape(-1,multiplier.shape[-1])
        result=torch.mm(flat,weight,out_dtype=torch.float32)
        return result.reshape(*multiplier.shape[:-1],weight.shape[1])
