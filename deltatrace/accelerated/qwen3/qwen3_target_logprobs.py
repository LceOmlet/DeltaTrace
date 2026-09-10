"""Original native per-row target log-probs in bounded storage; original sums stay outside."""
import torch
TARGET_ROWS=64

def target_logprobs(logits,target):
    assert logits.ndim==3 and logits.shape[:2]==target.shape
    rows16=[];rows32=[]
    for start in range(0,target.shape[1],TARGET_ROWS):
        z=logits[:,start:start+TARGET_ROWS];t=target[:,start:start+TARGET_ROWS,None]
        rows16.append(z.log_softmax(-1).gather(2,t).squeeze(-1))
        rows32.append(z.float().log_softmax(-1).gather(2,t).squeeze(-1))
    return torch.cat(rows16,1),torch.cat(rows32,1)
