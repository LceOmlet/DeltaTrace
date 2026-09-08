"""Batched FT hop bookkeeping, following pinned e81b3be core.py740-840.

Only scalar/vector recurrence; real native content contributions are supplied
by the caller. This is neither a model nor an attention/backward implementation.
The caller must independently validate against the unchanged author controller.
"""
import torch


class FTHopBatch:
    def __init__(self,lengths,sink_spans,thinking_spans,dtype=torch.bfloat16):
        self.lengths=tuple(lengths);self.width=max(lengths);self.dtype=dtype
        self.sinks=tuple(sink_spans);self.thinking=tuple(thinking_spans)
        assert len(self.sinks)==len(self.thinking)==len(lengths)
        self.mask=torch.zeros((len(lengths),self.width),dtype=torch.float32)
        self.weights=torch.zeros_like(self.mask);self.ratios=[1.0]*len(lengths)
        self.observation=torch.zeros_like(self.mask);self.hop=0;self.history=[]
        for b,(n,(s,e),(lo,hi)) in enumerate(zip(lengths,self.sinks,self.thinking)):
            assert 0<=lo<=hi<s<=e<n
            # Same default observation mask as author; final eligible-token
            # projection is separate and never changes thinking-ratio mass.
            self.mask[b,:n]=1;self.mask[b,lo:hi+1]=0;self.mask[b,s:e+1]=0
            self.mask[b,hi+1:]=0;self.weights[b,s:e+1]=1

    def consume(self,total):
        total=total.detach().to(device='cpu',dtype=torch.float32)
        assert total.shape==self.weights.shape and torch.isfinite(total).all()
        before=list(self.ratios);input_weights=self.weights.clone()
        for b,(n,(lo,hi)) in enumerate(zip(self.lengths,self.thinking)):
            assert torch.count_nonzero(total[b,n:])==0
            self.observation[b]+=total[b]*self.mask[b]*before[b]
            w=total[b,lo:hi+1].clone().to(self.dtype)
            denom=float(total[b,:n].sum());ratio=float(w.sum())/(denom+1e-12) if denom>0 else 0.0
            self.ratios[b]=(1.0 if self.hop==0 else before[b])*ratio
            self.weights[b].zero_();self.weights[b,lo:hi+1]=(w/(w.sum()+1e-12)).float()
        result={'hop':self.hop,'input_weights':input_weights,'token_total':total.clone(),
            'observation_sum':self.observation.clone(),'ratio_before':before,'ratio_after':list(self.ratios),
            'next_weights':self.weights.clone()}
        self.history.append(result);self.hop+=1;return result
