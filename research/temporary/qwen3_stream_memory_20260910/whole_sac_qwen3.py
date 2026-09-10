"""Public SAC on original native projections; no numeric rule replacement."""
from whole_remat_sac_base import WholeProgram,WholeRematFiniteGraphQwen3,root_endpoints,LiveScoresPacked
from native_projection_sac import NativeProjectionReuse
from qwen3_graph_validation import GraphValidation
from qwen_signed_secant_paired_public_fa import EndpointReplay
from remat_qwen3_graph import RematViews
from remat_finite_release import propagate_signed_secant


class SACProgram(WholeProgram):
    def invoke(self):
        self.inputs=None
        n=self.ids.shape[1]
        selected=[] if n<=300 else ['down'] if n<=700 else ['q','k','v','o','down']
        validation=GraphValidation()
        with NativeProjectionReuse(self.model,selected) as reuse:
            endpoints,scores=root_endpoints(self.model,self.ids,self.prompt_len)
            before,after=endpoints;self.inputs={'before':before,'after':after}
            master=before['paired_checkpoint'];assert master is after['paired_checkpoint']
            reuse.validation=validation
            paired=RematViews(self.model,master,validation)
            left=dict(before);right=dict(after);left['layers']=EndpointReplay(paired,0);right['layers']=EndpointReplay(paired,1)
            def release(li,variant):
                if li!=0:
                    paired.values=None;paired.index=None
                    paired.native.last_values=None;paired.native.last_index=None
            try:
                result=propagate_signed_secant(self.model,left,right,'rescale',progress=release,pv_rule='content_P1',finite_attention=self.finite,validation=validation)
                native=paired.native;assert native.calls==native.auxiliary_attention_calls==36
                result.update(extra_native_fa_attention_calls=36,extra_native_fa_attention_endpoint_trajectories=72,
                    public_FA_capture_checks=native.public_capture_checks,native_layer_replay_calls=36,
                    native_layer_replay_endpoint_trajectories=72,native_transformer_body_replay_equivalents=2,
                    native_layer_boundary_checks={'paired_batch':native.boundary_checks},native_paired_replay_layouts=paired.layout_records,
                    checkpoint_retained_tensor_bytes=master['tensor_bytes'],native_projection_SAC=reuse.activity,
                    recomputation_scope='Original native B2 model root and36 native B2 decoder calls in each whole graph. Selected MM outputs reused only within the same call through original public PyTorch SAC; other native operations recomputed. All finite operators and829 original strict predicates retained;36 extra SAC input predicates when selected. Root, copies, SAC, checks and full CPU return charged.')
                for row in result['native_fa_operand_audits']:
                    row['extra_native_attention_calls']=1
                    row['extra_call_count_scope']='One shared B2 public call per layer; both endpoint audit rows refer to this call.'
                packed=validation.finish(result)
            finally:paired.clear()
        return LiveScoresPacked(packed,scores)


class SACFiniteGraphQwen3(WholeRematFiniteGraphQwen3):
    program_type=SACProgram
