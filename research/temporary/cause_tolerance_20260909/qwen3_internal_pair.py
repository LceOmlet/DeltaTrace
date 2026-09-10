"""Original public FA paired checkpoint/replay plus separately named finite op."""
from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair,PairedReplayViews as FrozenPairedReplayViews,EndpointReplay
from qwen3_deferred_replay import NativeLayerReplay
from deferred_validation import DeferredValidation

class PairedReplayViews(FrozenPairedReplayViews):
    def __init__(self,model,master,activity=None,validation=None):
        self.native=NativeLayerReplay(model,master,activity=activity,validation=validation)
        self.index=None;self.values=None;self.layout_records=[]
from qwen3_internal_finite import propagate_signed_secant

def propagate_paired_secant(model,before,after,progress=None,pv_rule='content_P1',activity=None,finite_attention=None,finite_activity=None,observer=None):
    assert before['paired_checkpoint'] is after['paired_checkpoint']
    assert before['endpoint_index']==0 and after['endpoint_index']==1
    validation=DeferredValidation()
    master=before['paired_checkpoint'];paired=PairedReplayViews(model,master,activity=activity,validation=validation)
    left=dict(before);right=dict(after)
    left['layers']=EndpointReplay(paired,0);right['layers']=EndpointReplay(paired,1)
    try:
        result=propagate_signed_secant(model,left,right,'rescale',progress=progress,pv_rule=pv_rule,
            finite_attention=finite_attention,finite_activity=finite_activity,validation=validation,observer=observer)
        assert paired.native.calls==len(model.model.layers)==36
        assert paired.native.auxiliary_attention_calls==paired.native.calls
        result['extra_native_fa_attention_calls']=paired.native.auxiliary_attention_calls
        result['extra_native_fa_attention_endpoint_trajectories']=2*paired.native.auxiliary_attention_calls
        result['public_FA_capture_checks']=paired.native.public_capture_checks
        for audit in result['native_fa_operand_audits']:
            audit['extra_native_attention_calls']=1
            audit['extra_call_count_scope']='One shared B2 public call per layer; both endpoint audit rows refer to this call.'
        result['native_layer_replay_calls']=paired.native.calls
        result['native_layer_replay_endpoint_trajectories']=2*paired.native.calls
        result['native_transformer_body_replay_equivalents']=2
        result['native_layer_boundary_checks']={'paired_batch':paired.native.boundary_checks}
        result['native_paired_replay_layouts']=paired.layout_records
        result['checkpoint_retained_tensor_bytes']=master['tensor_bytes']
        result['recomputation_scope']='One actual default FA B2 root and36 B2 decoder replays, plus36 unchanged public B2 FA metadata calls. Testing matrix never used. Finite attribution is a separate traceable operator, not native backward.'
        return validation.finish(result)
    finally:paired.clear()
