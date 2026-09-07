"""Fuse an original FP32 difference into the complete FP64 audit dot.

This helper changes only numerical accounting implementation. It never supplies
an attribution multiplier or modifies model/autograd/FlashAttention behavior.
"""
import torch


def original_fp64_difference_dot(multiplier,after,before):
    difference=after-before
    return (multiplier.double()*difference.double()).sum()


compiled_fp64_difference_dot=torch.compile(original_fp64_difference_dot,fullgraph=True,dynamic=True,backend='inductor')


def audit_difference(after,before):
    assert after.dtype==before.dtype==torch.float32
    assert after.shape==before.shape
    return after,before
