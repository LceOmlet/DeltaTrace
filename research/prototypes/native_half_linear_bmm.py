"""Use existing PyTorch bmm for a one-element batch; no custom GEMM/backend.

The upstream compiler issue for mm(out_dtype) and bmm workaround is recorded at
https://github.com/pytorch/pytorch/issues/163275 . Input/accumulation/output
precision intent is unchanged: FP16 operands and FP32 output. Actual vendor
dispatch and numerical effects must be checked, not assumed equivalent bitwise.
"""
import torch


def native_half_linear_bmm(multiplier,weight):
    assert multiplier.dtype==torch.float32 and weight.dtype==torch.float16
    assert multiplier.shape[-1]==weight.shape[0] and weight.ndim==2
    flat=multiplier.reshape(-1,multiplier.shape[-1]).to(dtype=weight.dtype)
    result=torch.bmm(flat.unsqueeze(0),weight.unsqueeze(0),out_dtype=torch.float32).squeeze(0)
    return result.reshape(*multiplier.shape[:-1],weight.shape[1])
