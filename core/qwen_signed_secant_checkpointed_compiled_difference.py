"""Native layer-state checkpointing for the frozen independent signed secant rule.

Two original root forwards retain actual layer inputs and target outputs.
Each reverse visit calls the original Qwen decoder layer once per endpoint,
captures its real tensors with nonmutating hooks, and discards the prior layer.
No surrogate layer and no native VJP. Native replay calls are explicit costs.
"""
import time
from qwen_signed_secant_compiled_difference import propagate_signed_secant as full_secant_pullback


def capture_checkpoint_endpoint(model,ids,mask,prompt_len):
    import torch
    assert not model.training and mask.eq(1).all()
    assert model.config._attn_implementation=='flash_attention_2'
    assert all(layer.self_attn.sliding_window is None for layer in model.model.layers)
    assert model.config.hidden_act=='silu' and not model.config.attention_bias
    cache={'layers':[{} for _ in model.model.layers]}
    handles=[];original_arguments={}
    def copy(value):
        if isinstance(value,torch.Tensor):
            return value.detach().clone(memory_format=torch.preserve_format)
        if isinstance(value,tuple):
            return tuple(copy(x) for x in value)
        if isinstance(value,dict):
            return {k:copy(v) for k,v in value.items()}
        assert value is None or isinstance(value,(bool,int,float,str))
        return value
    def same(a,b):
        if a is b:
            return True
        if isinstance(a,torch.Tensor):
            return torch.equal(a,b)
        if isinstance(a,tuple):
            return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    def layer_input_recorder(index):
        def record(_module,args,kwargs):
            assert len(args)==1 and isinstance(args[0],torch.Tensor)
            cache['layers'][index]['x']=copy(args[0])
            assert kwargs.get('past_key_values') is None and not kwargs.get('use_cache',False)
            if not original_arguments:
                original_arguments.update(kwargs)
                cache['forward_kwargs']=copy(kwargs)
                cache['mask']=cache['forward_kwargs']['attention_mask']
                cache['cos'],cache['sin']=cache['forward_kwargs']['position_embeddings']
            else:
                assert set(kwargs)==set(original_arguments)
                assert all(same(value,original_arguments[key]) for key,value in kwargs.items())
        return record
    for index,layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(layer_input_recorder(index),with_kwargs=True))
    def final_norm_record(_module,args,output):
        cache['last']=copy(args[0]);cache['norm_out']=copy(output)
    handles.append(model.model.norm.register_forward_hook(final_norm_record))
    started=time.perf_counter()
    with torch.no_grad():
        try:
            result=model(input_ids=ids,attention_mask=mask,use_cache=False)
            logits=result.logits[0,prompt_len-1:-1];target=ids[0,prompt_len:]
            lp16=logits.log_softmax(-1).gather(1,target[:,None]).flatten()
            lp32=logits.float().log_softmax(-1).gather(1,target[:,None]).flatten()
            cache['score16']=float(lp16.sum());cache['score32_sum64']=float(lp32.double().sum())
            cache['target_logprobs32']=copy(lp32);cache['logits']=copy(logits);cache['target']=copy(target)
            cache['prompt_len']=prompt_len;cache['length']=ids.shape[1]
            del result,logits,lp16,lp32
        finally:
            for handle in handles:
                handle.remove()
    cache['capture_seconds']=time.perf_counter()-started
    seen=set()
    def bytes_of(value):
        if isinstance(value,torch.Tensor):
            if id(value) in seen:
                return 0
            seen.add(id(value));return value.numel()*value.element_size()
        if isinstance(value,dict):
            return sum(bytes_of(x) for x in value.values())
        if isinstance(value,(tuple,list)):
            return sum(bytes_of(x) for x in value)
        return 0
    cache['tensor_bytes']=bytes_of(cache)
    return cache


class NativeLayerReplay:
    def __init__(self,model,checkpoint,reference=None):
        self.model=model;self.checkpoint=checkpoint;self.reference=reference
        self.calls=0;self.checks=[];self.boundary_checks=[];self.last_index=None;self.last_values=None
    def __getitem__(self,index):
        import torch
        if index==self.last_index:
            return self.last_values
        layer=self.model.model.layers[index];values={};handles=[]
        def recorder(in_name=None,out_name=None):
            def record(_module,args,output):
                if in_name is not None:
                    values[in_name]=args[0].detach().clone(memory_format=torch.preserve_format)
                if out_name is not None:
                    values[out_name]=output.detach().clone(memory_format=torch.preserve_format)
            return record
        specs=[(layer.input_layernorm,'x','a'),(layer.self_attn.q_norm,'qpre','qnorm'),
            (layer.self_attn.k_norm,'kpre','knorm'),(layer.self_attn.v_proj,None,'v'),
            (layer.self_attn.o_proj,'concat','attn_out'),(layer.post_attention_layernorm,'mid','mlp_in'),
            (layer.mlp.gate_proj,None,'gate'),(layer.mlp.up_proj,None,'up'),
            (layer.mlp.down_proj,'product','mlp_out'),(layer,None,'out')]
        for module,in_name,out_name in specs:
            handles.append(module.register_forward_hook(recorder(in_name,out_name)))
        def attention_record(_module,_args,_kwargs,output):
            assert output[1] is None
            values['p']=None
        handles.append(layer.self_attn.register_forward_hook(attention_record,with_kwargs=True))
        import sys
        import flash_attn.flash_attn_interface as native_fa
        native_code=native_fa._flash_attn_forward.__code__;observed=[]
        def observer(frame,event,result):
            if event=='return' and frame.f_code is native_code:
                local=frame.f_locals
                assert local['dropout_p']==0 and local['causal'] and local['window_size']==(-1,-1)
                assert local['attn_mask'] is None and local['alibi_slopes'] is None and local['s_aux'] is None and local['softcap']==0
                for name in ['q','k','v']:
                    values['fa_'+name]=local[name].detach().clone(memory_format=torch.preserve_format)
                values['fa_out']=result[0].detach().clone(memory_format=torch.preserve_format)
                values['fa_lse']=result[5].detach().clone(memory_format=torch.preserve_format)
                observed.append(True)
        assert sys.getprofile() is None
        try:
            with torch.no_grad():
                self.calls+=1
                sys.setprofile(observer)
                layer(self.checkpoint['layers'][index]['x'],**self.checkpoint['forward_kwargs'])
                assert len(observed)==1
        finally:
            sys.setprofile(None)
            for handle in handles:
                handle.remove()
        expected_output=(self.checkpoint['layers'][index+1]['x']
            if index+1<len(self.model.model.layers) else self.checkpoint['last'])
        check={'layer':index,'native_input_exact':bool(torch.equal(values['x'],self.checkpoint['layers'][index]['x'])),
            'native_output_exact':bool(torch.equal(values['out'],expected_output))}
        self.boundary_checks.append(check)
        assert check['native_input_exact'] and check['native_output_exact'],f'Original decoder boundary replay mismatch at layer{index}'
        if self.reference is not None:
            expected=self.reference['layers'][index]
            assert set(values)==set(expected)
            checks={key:{'values_exact':bool(torch.equal(value,expected[key])),
                'strides_exact':value.stride()==expected[key].stride()} for key,value in values.items()}
            self.checks.append({'layer':index,'tensors':checks})
            assert all(x['values_exact'] for x in checks.values()),f'Native layer replay mismatch at layer{index}'
        self.last_index=index;self.last_values=values
        return values
    def clear(self):
        self.last_values=None;self.last_index=None


def propagate_checkpointed_secant(model,before,after,before_reference=None,after_reference=None,progress=None):
    import torch
    assert before['length']==after['length']
    first=NativeLayerReplay(model,before,before_reference)
    second=NativeLayerReplay(model,after,after_reference)
    left=dict(before);right=dict(after);left['layers']=first;right['layers']=second
    try:
        result=full_secant_pullback(model,left,right,'rescale',progress=progress)
        assert first.calls==second.calls==len(model.model.layers)
        result['native_layer_replay_calls']=first.calls+second.calls
        result['native_transformer_body_replay_equivalents']=2
        result['native_layer_replay_checks']={'before':first.checks,'after':second.checks}
        result['native_layer_boundary_checks']={'before':first.boundary_checks,'after':second.boundary_checks}
        result['checkpoint_retained_tensor_bytes']=before['tensor_bytes']+after['tensor_bytes']
        result['recomputation_scope']='Two actual native FA root forwards and72 original FA decoder replays. Zero auxiliary probability calls. P is an explicitly reconstructed attribution operand from actual QKV/LSE; original normalized-score finite-softmax rule and complete FP64 audits remain. No model/native-autograd replacement.'
        return result
    finally:
        first.clear();second.clear()
