"""Restore the author's epsilon formula only at FP16 zero/zero saved ratios."""
from contextlib import contextmanager

@contextmanager
def zero_ratio_guard():
    import torch
    import lrp_rules
    original=lrp_rules.IdentityRuleImplicitFn.forward
    receipt=dict(calls=0,repaired_zero_ratios=0)
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
    try:yield receipt
    finally:
        lrp_rules.IdentityRuleImplicitFn.forward=staticmethod(original)
        assert lrp_rules.IdentityRuleImplicitFn.forward is original
