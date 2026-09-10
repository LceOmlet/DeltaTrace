"""Retain actual native root operands instead of replaying Qwen3 decoders.

All finite mathematics is imported unchanged from the pinned retained path.
Storage grows linearly with layers and sequence length; no attention matrix
or cross-invocation cache is introduced. Public FA LSE calls remain charged.
"""
import inspect
import sys
import torch
from flash_attn import flash_attn_func
from qwen3_graph_validation import GraphValidation as DeferredValidation
from qwen_signed_secant_paired_public_fa import PairedReplayViews as _Views,EndpointReplay
from qwen3_graph_tensor_finite import propagate_signed_secant


from qwen3_root_retained import NativeRootTape


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
