"""Restore the author's epsilon formula only at FP16 zero/zero saved ratios."""
from contextlib import contextmanager

@contextmanager
def zero_ratio_guard(model=None):
    import torch
    import lrp_rules
    original=lrp_rules.IdentityRuleImplicitFn.forward
    receipt=dict(calls=0,repaired_zero_ratios=0)
    parameters={p.untyped_storage().data_ptr() for p in model.parameters()} if model is not None else set()
    def pack(tensor):
        if tensor.device.type=='cpu' or tensor.untyped_storage().data_ptr() in parameters:
            return ('resident',tensor.detach())
        assert tensor.layout==torch.strided
        # Copy the entire storage, then reconstruct the SAME strides and offset.
        # Generic save_on_cpu(pin_memory=True) makes saved views contiguous and
        # can select a different FP16 GEMM path during backward on this device.
        flat=tensor.detach().as_strided((tensor.untyped_storage().nbytes()//tensor.element_size(),),(1,),0)
        cpu=torch.empty(flat.shape,dtype=flat.dtype,device='cpu',pin_memory=True)
        cpu.copy_(flat,non_blocking=False)
        return ('offloaded',tensor.device,tensor.shape,tensor.stride(),tensor.storage_offset(),cpu)
    def unpack(payload):
        if payload[0]=='resident':return payload[1]
        _,device,shape,strides,offset,cpu=payload
        return cpu.to(device,non_blocking=False).as_strided(shape,strides,offset)
    def guarded(ctx,fn,input,epsilon=1e-10):
        output=fn(input)
        if input.requires_grad:
            ratio=output/(input+epsilon)
            bad=~torch.isfinite(ratio)
            count=int(bad.sum())
            receipt['calls']+=1
            if count:
                assert bool(((input[bad]==0)&(output[bad]==0)).all()),'Unexpected nonzero nonfinite LRP ratio'
                # Re-evaluate only affected elements with representable epsilon.
                repaired=output[bad].float()/(input[bad].float()+epsilon)
                assert bool(torch.isfinite(repaired).all()) and not bool(repaired.any())
                ratio=ratio.clone();ratio[bad]=repaired.to(ratio.dtype)
                receipt['repaired_zero_ratios']+=count
            ctx.save_for_backward(ratio)
        return output
    lrp_rules.IdentityRuleImplicitFn.forward=staticmethod(guarded)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
            yield receipt
    finally:
        lrp_rules.IdentityRuleImplicitFn.forward=staticmethod(original)
        assert lrp_rules.IdentityRuleImplicitFn.forward is original
