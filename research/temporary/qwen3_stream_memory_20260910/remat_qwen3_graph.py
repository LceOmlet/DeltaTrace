"""One native finite graph with per-layer original decoder rematerialization."""
import time
import torch
from qwen3_finite_graph import Program,FiniteGraphQwen3,slim
from qwen3_graph_storage_inputs import clone_tree,refresh
from qwen3_graph_validation import GraphValidation
from qwen3_graph_tensor_finite import propagate_signed_secant
from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair,PairedReplayViews,EndpointReplay
from remat_native_replay import GraphLayerReplay


class RematViews(PairedReplayViews):
    def __init__(self,model,master,validation):
        self.native=GraphLayerReplay(model,master,validation)
        self.index=None;self.values=None;self.layout_records=[]


class RematProgram(Program):
    def invoke(self):
        before=self.inputs['before'];after=self.inputs['after'];master=before['paired_checkpoint']
        assert master is after['paired_checkpoint']
        validation=GraphValidation();paired=RematViews(self.model,master,validation)
        left=dict(before);right=dict(after);left['layers']=EndpointReplay(paired,0);right['layers']=EndpointReplay(paired,1)
        try:
            result=propagate_signed_secant(self.model,left,right,'rescale',pv_rule='content_P1',finite_attention=self.finite,validation=validation)
            native=paired.native;assert native.calls==native.auxiliary_attention_calls==36
            result.update(extra_native_fa_attention_calls=36,extra_native_fa_attention_endpoint_trajectories=72,
                public_FA_capture_checks=native.public_capture_checks,native_layer_replay_calls=36,
                native_layer_replay_endpoint_trajectories=72,native_transformer_body_replay_equivalents=2,
                native_layer_boundary_checks={'paired_batch':native.boundary_checks},native_paired_replay_layouts=paired.layout_records,
                checkpoint_retained_tensor_bytes=master['tensor_bytes'],
                recomputation_scope='Fresh original B2 root, then36 original B2 decoder replays inside native graph.36 additional public B2 FA LSE calls; identical finite math and all strict checks. All root, replay, refresh and full CPU return costs charged.')
            for row in result['native_fa_operand_audits']:
                row['extra_native_attention_calls']=1
                row['extra_call_count_scope']='One shared B2 public call per layer; both endpoint audit rows refer to this call.'
            return validation.finish(result)
        finally:paired.clear()


class RematFiniteGraphQwen3(FiniteGraphQwen3):
    def attribute(self,before_ids,after_ids,mask,prompt_len,*,mutation_audit=False):
        assert not self.busy and not self.model.training
        self.busy=True
        try:
            torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            geometry=(tuple(after_ids.shape),prompt_len)
            if self.signature is not None and self.signature[:2]!=geometry:
                self.program=None;self.signature=None
            before,after=capture_checkpoint_pair(self.model,before_ids,after_ids,mask,prompt_len)
            root_done=time.perf_counter()
            assert before['length']==after['length'] and before['prompt_len']==after['prompt_len']
            assert torch.equal(before['target'],after['target']) and torch.equal(before['cos'],after['cos']) and torch.equal(before['sin'],after['sin'])
            assert before['mask'] is None and after['mask'] is None
            signature=(tuple(after_ids.shape),prompt_len,tuple(after['logits'].shape),str(after_ids.device),str(next(self.model.parameters()).dtype))
            source={'before':slim(before),'after':slim(after)}
            build=None
            if self.program is None:
                self.program=RematProgram(self.model,source,before['paired_checkpoint']['tensor_bytes'],self.finite)
                copy_checks=self.program.refresh(source,audit=mutation_audit)
                build=self.program.build();self.signature=signature
            else:
                assert signature==self.signature
                copy_checks=self.program.refresh(source,audit=mutation_audit)
            del source
            refreshed=time.perf_counter();result=self.program.replay(before,after,started)
            result['graph_input_copy_audit']={'enabled':mutation_audit,'predicates':copy_checks,'all_passed':True}
            result['native_graph_execution']={'graph_replays':1,'geometry_cache_entries':1,'static_input_tensors':len(self.program.paths),
                'static_input_storage_copies':len(self.program.storage_groups),'every_input_tensor_refreshed':True,'attribution_results_reused':False,
                'build_this_call':build,'finite_program_warmups_this_call':2 if build else 0,'native_model_root_calls':1,
                'native_model_root_endpoint_batch':2,'native_decoder_GPU_replays':36,'public_auxiliary_FA_GPU_operations_per_replay':36,
                'finite_FA_GPU_operations_per_replay':36,'native_decoder_Python_calls_this_call':108 if build else 0}
            result['rematerialized_root']={'enabled':True,'retained_native_operand_layers_between_root_and_finite':0,
                'host_sections_seconds':{'root_capture':root_done-started,'checks_copy_and_optional_build':refreshed-root_done,'replay_and_CPU_return':time.perf_counter()-refreshed},
                'timing_scope':'External synchronized complete attribution timer is authoritative. Cold includes2 finite+native decoder warmups,1 capture and1 replay.'}
            return result
        except BaseException:
            self.program=None;self.signature=None;raise
        finally:self.busy=False
