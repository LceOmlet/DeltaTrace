"""Retain actual native root operands instead of replaying Qwen3 decoders.

All finite mathematics is imported unchanged from the pinned retained path.
Storage grows linearly with layers and sequence length; no attention matrix
or cross-invocation cache is introduced. Public FA LSE calls remain charged.
"""
import inspect
import sys
import torch
from flash_attn import flash_attn_func
from batched_validation_rope import BatchedValidation as DeferredValidation
from qwen_signed_secant_paired_public_fa import PairedReplayViews as _Views,EndpointReplay
from qwen3_rope_finite import propagate_signed_secant


class NativeRootTape:
    def __init__(self,model,*,mutation_audit=False):
        self.model=model;self.mutation_audit=mutation_audit
        self.layers=[{} for _ in model.model.layers];self.arguments={};self.versions={};self.snapshots={}
        self.handles=[];self.active=None;self.root_calls=[0]*len(self.layers);self.fa_calls=[0]*len(self.layers)
        self.signature=inspect.signature(flash_attn_func);self.public_code=flash_attn_func.__code__
        self.storage={};self.maximum_retained_bytes=0;self.audit_predicates=0
        self._entered=False

    def retain(self,index,name,value):
        assert name not in self.layers[index],(index,name)
        tensor=value.detach();self.layers[index][name]=tensor
        self.versions[index,name]=tensor._version
        storage=tensor.untyped_storage();key=(str(tensor.device),storage.data_ptr())
        self.storage[key]=storage.nbytes()
        self.maximum_retained_bytes=sum(self.storage.values())
        if self.mutation_audit:self.snapshots[index,name]=tensor.clone(memory_format=torch.preserve_format)

    def observe_fa(self,frame,event,result):
        if event!='return' or frame.f_code is not self.public_code:return
        index=self.active;assert index is not None
        assert isinstance(result,torch.Tensor)
        local=frame.f_locals;arguments={}
        for name,parameter in self.signature.parameters.items():
            assert parameter.kind not in (parameter.VAR_POSITIONAL,parameter.VAR_KEYWORD)
            assert name in local;arguments[name]=local[name]
        assert arguments['dropout_p']==0 and arguments['causal'] and arguments['window_size']==(-1,-1)
        for name in ['attn_mask','alibi_slopes','s_aux']:assert arguments.get(name) is None
        assert arguments.get('softcap',0)==0 and not arguments['return_attn_probs'] and not arguments.get('return_max_logit',False)
        for name in ['q','k','v']:self.retain(index,'fa_'+name,local[name])
        self.retain(index,'fa_out',result);self.arguments[index]=arguments;self.fa_calls[index]+=1

    def __enter__(self):
        assert not self._entered and sys.getprofile() is None
        self._entered=True
        try:
            for index,layer in enumerate(self.model.model.layers):
                def start(_module,args,index=index):
                    assert self.active is None;self.active=index;self.root_calls[index]+=1
                def finish(_module,args,output,index=index):
                    assert self.active==index;self.active=None
                self.handles.append(layer.register_forward_pre_hook(start))
                specs=[(layer.input_layernorm,'x','a'),(layer.self_attn.q_norm,'qpre','qnorm'),
                  (layer.self_attn.k_norm,'kpre','knorm'),(layer.self_attn.v_proj,None,'v'),
                  (layer.self_attn.o_proj,'concat','attn_out'),(layer.post_attention_layernorm,'mid','mlp_in'),
                  (layer.mlp.gate_proj,None,'gate'),(layer.mlp.up_proj,None,'up'),
                  (layer.mlp.down_proj,'product','mlp_out'),(layer,None,'out')]
                for module,in_name,out_name in specs:
                    def record(_module,args,output,index=index,in_name=in_name,out_name=out_name):
                        assert self.active==index
                        if in_name is not None:self.retain(index,in_name,args[0])
                        if out_name is not None:self.retain(index,out_name,output)
                    self.handles.append(module.register_forward_hook(record))
                def attention(_module,args,kwargs,output,index=index):
                    assert output[1] is None;self.layers[index]['p']=None
                self.handles.append(layer.self_attn.register_forward_hook(attention,with_kwargs=True))
                self.handles.append(layer.register_forward_hook(finish,always_call=True))
            sys.setprofile(self.observe_fa)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info());raise

    def __exit__(self,*args):
        sys.setprofile(None)
        for handle in self.handles:handle.remove()
        self.handles.clear();self._entered=False
        if args[0] is None:
            assert self.active is None and all(n==1 for n in self.root_calls) and all(n==1 for n in self.fa_calls)

    def clear(self):
        self.layers.clear();self.arguments.clear();self.versions.clear();self.snapshots.clear();self.storage.clear()


class NativeRootAccess:
    def __init__(self,model,master,tape,validation,activity):
        self.model=model;self.master=master;self.tape=tape;self.validation=validation
        self.activity=activity if activity is not None else {'auxiliary_attempts':0,'auxiliary_completed':0,'metadata':[]}
        self.public_capture_checks=[];self.boundary_checks=[];self.auxiliary_attention_calls=0
        self.served_layers=0;self.last_index=None;self.last_values=None

    def __getitem__(self,index):
        if index==self.last_index:return self.last_values
        values=self.tape.layers[index];assert values
        self.tape.layers[index]={};self.served_layers+=1
        for name,value in values.items():
            if value is None:continue
            assert value._version==self.tape.versions.pop((index,name)),(index,name)
            if self.tape.mutation_audit:
                self.validation.equal(value,self.tape.snapshots.pop((index,name)),f'layer{index}.root_retention_{name}')
                self.tape.audit_predicates+=1
        arguments=self.tape.arguments.pop(index);arguments['return_attn_probs']=True
        with torch.no_grad():
            self.auxiliary_attention_calls+=1;self.activity['auxiliary_attempts']+=1
            auxiliary=flash_attn_func(**arguments);self.activity['auxiliary_completed']+=1
        assert isinstance(auxiliary,tuple) and len(auxiliary)==3
        output,lse,testing=auxiliary
        descriptor={'type':type(testing).__name__,'shape':list(testing.shape) if isinstance(testing,torch.Tensor) else None,
          'numel':testing.numel() if isinstance(testing,torch.Tensor) else None}
        assert testing is None or isinstance(testing,torch.Tensor) and testing.numel()==0,descriptor
        self.activity['metadata'].append({'layer':index,'testing_return':descriptor})
        assert lse.shape==(values['fa_q'].shape[0],values['fa_q'].shape[2],values['fa_q'].shape[1]) and lse.dtype==torch.float32
        self.validation.finite(lse,f'layer{index}.public_lse')
        exact=self.validation.equal(output,values['fa_out'],f'layer{index}.public_output')
        values['fa_lse']=lse.detach()
        self.public_capture_checks.append({'layer':index,'public_output_exact_to_actual_model_FA':exact,
          'testing_matrix_numel':0,'testing_return':descriptor,'auxiliary_batch_size':values['fa_q'].shape[0],
          'lse_shape':list(lse.shape),'lse_dtype':str(lse.dtype),'public_function_module':flash_attn_func.__module__,
          'private_FA_slots_read':False,'model_output_replaced':False,'auxiliary_public_FA_calls':1})
        expected=self.master['layers'][index+1]['x'] if index+1<len(self.model.model.layers) else self.master['last']
        self.boundary_checks.append({'layer':index,
          'native_input_exact':self.validation.equal(values['x'],self.master['layers'][index]['x'],f'layer{index}.root_input'),
          'native_output_exact':self.validation.equal(values['out'],expected,f'layer{index}.root_output')})
        self.last_index=index;self.last_values=values
        return values

    def clear(self):self.last_values=None;self.last_index=None;self.tape.clear()


class PairedRootViews(_Views):
    def __init__(self,model,master,tape,validation,activity):
        self.native=NativeRootAccess(model,master,tape,validation,activity)
        self.index=None;self.values=None;self.layout_records=[]


def propagate_root_tape(model,before,after,tape,*,progress=None,pv_rule='content_P1',activity=None,finite_attention=None,finite_activity=None):
    assert before['paired_checkpoint'] is after['paired_checkpoint']
    assert before['endpoint_index']==0 and after['endpoint_index']==1
    validation=DeferredValidation();master=before['paired_checkpoint']
    paired=PairedRootViews(model,master,tape,validation,activity)
    left=dict(before);right=dict(after);left['layers']=EndpointReplay(paired,0);right['layers']=EndpointReplay(paired,1)
    try:
        result=propagate_signed_secant(model,left,right,'rescale',progress=progress,pv_rule=pv_rule,
          finite_attention=finite_attention,finite_activity=finite_activity,validation=validation)
        assert paired.native.served_layers==paired.native.auxiliary_attention_calls==len(model.model.layers)==36
        result.update(extra_native_fa_attention_calls=paired.native.auxiliary_attention_calls,
          extra_native_fa_attention_endpoint_trajectories=2*paired.native.auxiliary_attention_calls,
          public_FA_capture_checks=paired.native.public_capture_checks,native_layer_replay_calls=0,
          native_layer_replay_endpoint_trajectories=0,native_transformer_body_replay_equivalents=0,
          native_layer_boundary_checks={'paired_root':paired.native.boundary_checks},
          native_paired_root_layouts=paired.layout_records,checkpoint_retained_tensor_bytes=master['tensor_bytes'],
          root_operand_unique_storage_bytes=tape.maximum_retained_bytes,
          root_retention_mutation_audit={'enabled':tape.mutation_audit,'predicates':tape.audit_predicates},
          recomputation_scope='One actual default-FA B2 root; retain native intermediates and perform no decoder replay.36 additional unchanged public B2 FA calls obtain LSE. Finite propagation and strict diagnostics remain.')
        for audit in result['native_fa_operand_audits']:
            audit['extra_native_attention_calls']=1
            audit['extra_call_count_scope']='One shared B2 public call per layer; both endpoint audit rows refer to this call.'
        return validation.finish(result)
    finally:paired.clear()
