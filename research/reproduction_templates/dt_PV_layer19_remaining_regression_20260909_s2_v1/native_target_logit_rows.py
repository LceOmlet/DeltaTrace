"""Select target predictor rows via the original model's logits_to_keep API.

No vocabulary selection, logits reconstruction or alternate output head. The
union handles distinct per-example target positions; packing preserves the
existing target/endpoint order expected by PackedAnswerTargets and finite seed.
"""
import torch


class NativeTargetLogitRows:
    def __init__(self,selection):
        self.selection=selection
        self.rows,inverse=torch.unique(selection.positions,sorted=True,return_inverse=True)
        self.packed_rows=inverse.repeat_interleave(2)
        if not torch.equal(self.rows[inverse],selection.positions):
            raise ValueError('Native output rows do not reproduce the target predictor positions.')

    def pack_logits(self,actual_selected_logits):
        if actual_selected_logits.ndim!=3 or actual_selected_logits.shape[:2]!=(2*self.selection.batch,len(self.rows)):
            raise ValueError('Expected native logits for all endpoints and selected predictor rows.')
        return actual_selected_logits[self.selection.paired_samples,self.packed_rows].contiguous()
