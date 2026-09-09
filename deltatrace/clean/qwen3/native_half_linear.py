"""Native library FP16-input/FP32-output attribution projections.

Only the multiplier is rounded to the existing model's FP16 weight dtype.
This is an explicit numerical attribution variant, not a native model gradient.
No model forward, autograd replacement or custom GEMM implementation.
"""
import torch


def native_half_linear(multiplier,weight):
    assert multiplier.dtype==torch.float32 and weight.dtype==torch.float16
    assert multiplier.shape[-1]==weight.shape[0] and weight.ndim==2
    with torch.profiler.record_function('ATTR_NATIVE_HALF_LINEAR'):
        flat=multiplier.reshape(-1,multiplier.shape[-1]).to(dtype=weight.dtype)
        result=torch.mm(flat,weight,out_dtype=torch.float32)
        return result.reshape(*multiplier.shape[:-1],weight.shape[1])
