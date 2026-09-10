"""Bound only independent target rows; keep vocabulary reductions and head MM intact."""
import torch
from qwen3_projection_cast_boundaries import seed_half
TARGET_ROWS=64

def target_logprobs(logits,target):
    assert logits.ndim==3 and logits.shape[:2]==target.shape
    rows16=[];rows32=[]
    for start in range(0,target.shape[1],TARGET_ROWS):
        z=logits[:,start:start+TARGET_ROWS];t=target[:,start:start+TARGET_ROWS,None]
        rows16.append(z.log_softmax(-1).gather(2,t).squeeze(-1))
        rows32.append(z.float().log_softmax(-1).gather(2,t).squeeze(-1))
    return torch.cat(rows16,1),torch.cat(rows32,1)

def target_seed(z0,z1,target):
    assert z0.ndim==z1.ndim==2 and z0.shape==z1.shape and target.shape==z0.shape[:1]
    # Write identical per-row seeds into one contiguous matrix so the following
    # original native FP16-input GEMM has exactly its original shape and strides.
    seed=torch.empty_like(z0,dtype=torch.float16);checks=[]
    for start in range(0,target.shape[0],TARGET_ROWS):
        value,check=seed_half(z0[start:start+TARGET_ROWS].float(),z1[start:start+TARGET_ROWS].float(),target[start:start+TARGET_ROWS])
        seed[start:start+TARGET_ROWS].copy_(value);checks.append(check)
        del value
    return seed,torch.stack(checks).all()
