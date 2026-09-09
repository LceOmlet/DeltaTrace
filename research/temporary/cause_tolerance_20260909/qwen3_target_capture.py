"""Temporary native Qwen3 target-logit capture, copied from the frozen capturer.

Only the root logits_to_keep argument and output slice differ. This module does
not implement a model, attention, or backward. The original checkpoint, decoder
replay and finite propagation contracts are retained. Sample B1, endpoint E2.
"""
import time

def capture_checkpoint_pair_raw(model,ids,mask,prompt_len):
    import torch
    assert ids.shape[0]==mask.shape[0]==2
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
            target_count=ids.shape[1]-prompt_len
            assert target_count>0
            result=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=target_count+1)
            assert result.logits.shape[1]==target_count+1
            logits=result.logits[:,:-1];target=ids[:,prompt_len:]
            lp16=logits.log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            lp32=logits.float().log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            cache['score16']=lp16.sum(-1).cpu().tolist();cache['score32_sum64']=lp32.double().sum(-1).cpu().tolist()
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


def capture_checkpoint_pair(model,before_ids,after_ids,mask,prompt_len):
    import torch
    assert before_ids.shape==after_ids.shape==mask.shape and before_ids.shape[0]==1
    assert torch.equal(before_ids[:,prompt_len:],after_ids[:,prompt_len:])
    ids=torch.cat((before_ids,after_ids),0);masks=torch.cat((mask,mask),0)
    master=capture_checkpoint_pair_raw(model,ids,masks,prompt_len)
    # The actual model creates shared position_ids [1,N] and RoPE [1,N,D].
    # Keep those actual shared tensors; do not split or recompute them.
    assert master['mask'] is None
    assert master['cos'].shape[0]==master['sin'].shape[0]==1
    endpoints=[]
    for endpoint in range(2):
        view={k:master[k] for k in ['cos','sin','mask','length','prompt_len']}
        view['layers']=[{'x':layer['x'][endpoint:endpoint+1]} for layer in master['layers']]
        for key in ['last','norm_out']:view[key]=master[key][endpoint:endpoint+1]
        for key in ['logits','target','target_logprobs32']:view[key]=master[key][endpoint]
        for key in ['score16','score32_sum64']:view[key]=master[key][endpoint]
        view['paired_checkpoint']=master;view['endpoint_index']=endpoint
        endpoints.append(view)
    return tuple(endpoints)
