"""Compile only the original FP64 audit dot; never a model or attribution multiplier."""
import torch

def original_fp64_dot(a,b):
    return (a.double()*b.double()).sum()

compiled_fp64_dot=torch.compile(original_fp64_dot,fullgraph=True,dynamic=True,backend='inductor')
