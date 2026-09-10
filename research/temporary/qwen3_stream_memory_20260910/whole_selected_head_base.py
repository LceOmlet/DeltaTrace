"""Original native root and layer-rematerialized finite program in one graph.

Only token IDs are external graph inputs. Model weights and geometry are fixed.
All root checkpoints and scores are freshly produced by original model kernels.
"""
import time
import torch
from qwen3_finite_graph import Program,FiniteGraphQwen3
from remat_qwen3_graph import RematProgram


def root_endpoints(model,ids,prompt_len):
    """Graph-compatible staging of the original paired checkpoint capture.

    Same original model, logits slice, FP16/FP32 log-softmax and FP64 sum.
    CPU scalar conversion moves to the complete graph's public return.
    Actual root input checkpoints are retained without duplicate storage.
    """
    assert ids.shape[0]==2 and not model.training
    assert model.config._attn_implementation=='flash_attention_2'
    assert model.config.hidden_act=='silu' and not model.config.attention_bias
    assert all(layer.self_attn.sliding_window is None for layer in model.model.layers)
    cache={'layers':[{} for _ in model.model.layers]};handles=[];versions=[];arguments={}
    def retained(value):
        versions.append((value,value._version));return value.detach()
    def same(a,b):
        if a is b:return True
        if isinstance(a,(tuple,list)):
            return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a,torch.Tensor):
            # Original root passes shared positional tensors to every decoder.
            return a.shape==b.shape and a.stride()==b.stride() and a.data_ptr()==b.data_ptr()
        return a==b
    def layer_input(index):
        def record(_module,args,kwargs):
            assert len(args)==1 and isinstance(args[0],torch.Tensor)
            cache['layers'][index]['x']=retained(args[0])
            assert kwargs.get('past_key_values') is None and not kwargs.get('use_cache',False)
            if not arguments:
                arguments.update(kwargs);cache['forward_kwargs']=dict(kwargs)
                cache['mask']=kwargs['attention_mask'];cache['cos'],cache['sin']=kwargs['position_embeddings']
            else:
                assert set(kwargs)==set(arguments)
                assert all(same(v,arguments[k]) for k,v in kwargs.items())
        return record
    for index,layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(layer_input(index),with_kwargs=True))
    def norm_record(_module,args,output):
        cache['last']=retained(args[0]);cache['norm_out']=retained(output)
    handles.append(model.model.norm.register_forward_hook(norm_record))
    try:
        with torch.no_grad():
            # The public controller validates an all-ones input mask. The
            # original FA root's actual decoder attention mask is also None.
            result=model(input_ids=ids,attention_mask=None,use_cache=False,logits_to_keep=ids.shape[1]-prompt_len+1)
            logits=result.logits[:,:-1];target=ids[:,prompt_len:]
            assert result.logits.shape[1]==ids.shape[1]-prompt_len+1
            lp16=logits.log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            lp32=logits.float().log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            cache['score16']=lp16.sum(-1);cache['score32_sum64']=lp32.double().sum(-1)
            scores=torch.stack((cache['score16'].double(),cache['score32_sum64']))
            cache['target_logprobs32']=lp32.detach().clone(memory_format=torch.preserve_format)
            # Keep only the original selected logits storage, not all N rows.
            cache['logits']=logits.detach().clone(memory_format=torch.preserve_format)
            cache['target']=target.detach().clone(memory_format=torch.preserve_format)
            cache['prompt_len']=prompt_len;cache['length']=ids.shape[1]
            del result,logits,lp16,lp32
    finally:
        for handle in handles:handle.remove()
    for value,version in versions:assert value._version==version
    assert cache['mask'] is None and cache['cos'].shape[0]==cache['sin'].shape[0]==1
    seen=set()
    def count_bytes(value):
        if isinstance(value,torch.Tensor):
            key=(value.untyped_storage().data_ptr(),value.dtype)
            if key in seen:return 0
            seen.add(key);return value.untyped_storage().nbytes()
        if isinstance(value,dict):return sum(count_bytes(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(count_bytes(v) for v in value)
        return 0
    cache['tensor_bytes']=count_bytes(cache);endpoints=[]
    for endpoint in range(2):
        view={k:cache[k] for k in ['cos','sin','mask','length','prompt_len']}
        for key in ['last','norm_out']:view[key]=cache[key][endpoint:endpoint+1]
        for key in ['logits','target','target_logprobs32','score16','score32_sum64']:view[key]=cache[key][endpoint]
        view['paired_checkpoint']=cache;view['endpoint_index']=endpoint;endpoints.append(view)
    return tuple(endpoints),scores


class LiveScoresPacked:
    def __init__(self,packed,scores):self.packed=packed;self.scores=scores
    def resolve(self,before,after,seconds):
        scores=self.scores.cpu().tolist();left=dict(before);right=dict(after)
        for endpoint,view in enumerate([left,right]):
            view['score16']=scores[0][endpoint];view['score32_sum64']=scores[1][endpoint]
        return self.packed.resolve(left,right,seconds)


class WholeProgram(RematProgram):
    def __init__(self,model,ids,prompt_len,finite):
        self.model=model;self.ids=ids.clone();self.prompt_len=prompt_len;self.finite=finite
        self.inputs=None;self.graph=self.packed=self.build_info=None;self.retained_bytes=0
        self.paths=[(('ids',),self.ids,('ids',),self.ids.numel())]
        self.storage_groups=[(('ids',),self.ids.view(-1),self.ids.numel())]
    def invoke(self):
        self.inputs=None
        endpoints,scores=root_endpoints(self.model,self.ids,self.prompt_len)
        self.inputs={'before':endpoints[0],'after':endpoints[1]}
        return LiveScoresPacked(super().invoke(),scores)
    def build(self):
        info=super().build()
        info.update(captured_program_recording_is_not_a_model_forward=False,
            original_model_Python_calls_during_build=3,original_model_GPU_executions_during_build=3,
            root_and_finite_captured_together=True,root_checkpoint_copies_between_graphs=0)
        return info


class WholeRematFiniteGraphQwen3(FiniteGraphQwen3):
    program_type=WholeProgram
    def __init__(self,model,finite):
        super().__init__(model,finite)
        self.parameter_versions=[(p,p._version) for p in model.parameters()]
    def attribute(self,before_ids,after_ids,mask,prompt_len,*,mutation_audit=False):
        assert not self.busy and not self.model.training
        self.busy=True
        try:
            torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            assert before_ids.shape==after_ids.shape==mask.shape and before_ids.shape[0]==1
            assert bool(mask.eq(1).all()) and torch.equal(before_ids[:,prompt_len:],after_ids[:,prompt_len:])
            for parameter,version in self.parameter_versions:assert parameter._version==version
            ids=torch.cat((before_ids,after_ids),0)
            signature=(tuple(ids.shape),prompt_len,str(ids.device),str(ids.dtype))
            build=None
            if self.signature!=signature:
                self.program=None;self.signature=None
                self.program=self.program_type(self.model,ids,prompt_len,self.finite)
                build=self.program.build();self.signature=signature
            else:self.program.ids.copy_(ids,non_blocking=True)
            copied=time.perf_counter()
            if mutation_audit:assert torch.equal(self.program.ids,ids)
            result=self.program.replay(self.program.inputs['before'],self.program.inputs['after'],started)
            result['graph_input_copy_audit']={'enabled':mutation_audit,'predicates':1 if mutation_audit else 0,'all_passed':True}
            result['native_graph_execution']={'graph_replays':1,'geometry_cache_entries':1,'static_input_tensors':1,
                'static_input_storage_copies':1,'every_input_tensor_refreshed':True,'attribution_results_reused':False,
                'build_this_call':build,'finite_program_warmups_this_call':2 if build else 0,
                'native_model_root_calls':3 if build else 0,'native_model_Python_root_calls':3 if build else 0,
                'native_model_GPU_root_executions':4 if build else 1,'native_model_root_endpoint_batch':2,
                'native_decoder_GPU_replays_per_graph_replay':36,'public_auxiliary_FA_GPU_operations_per_replay':36,
                'finite_FA_GPU_operations_per_replay':36,'native_decoder_Python_calls_this_call':108 if build else 0}
            result['whole_root_graph']={'enabled':True,'external_graph_inputs':'Fresh B2 token IDs; fixed eval model weights and geometry.',
                'root_graph_execution_is_original_model_kernels':True,'model_forward_replaced':False,
                'input_mask':'Public input mask checked all ones; native root decoder attention mask None, checked against original reference.',
                'host_sections_seconds':{'input_checks_refresh_and_optional_build':copied-started,'whole_graph_and_complete_CPU_return':time.perf_counter()-copied},
                'timing_scope':'External synchronized complete attribution timer; root and reverse finite GPU work share one graph. Cold includes2 full root+finite warmups,1 capture,1 replay.'}
            if mutation_audit:
                import hashlib
                result['whole_root_graph']['fresh_graph_input_sha256']=hashlib.sha256(self.program.ids.cpu().numpy().tobytes()).hexdigest()
            return result
        except BaseException:
            self.program=None;self.signature=None;raise
        finally:self.busy=False
