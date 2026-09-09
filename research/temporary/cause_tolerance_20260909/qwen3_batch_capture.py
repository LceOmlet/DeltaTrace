"""Native root checkpoint collector with explicit packed target rows."""
import time

def capture_checkpoint_batch_raw(model,ids,mask,target_samples,target_positions,target_ids):
    import torch
    assert ids.shape==mask.shape and ids.shape[0]%2==0
    batch=ids.shape[0]//2
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
            paired_samples=2*target_samples[None,:]+torch.arange(2,device=ids.device)[:,None]
            logits=result.logits[paired_samples,target_positions[None,:]]
            target=target_ids[None,:].expand(2,-1)
            lp16=logits.log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            lp32=logits.float().log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            cache['score16']=lp16.sum(-1).cpu().tolist();cache['score32_sum64']=lp32.double().sum(-1).cpu().tolist()
            cache['scores_by_sample']=[torch.zeros(batch,device=ids.device,dtype=torch.float64).index_add_(0,target_samples,v.double()).cpu().tolist() for v in lp32]
            cache['target_logprobs32']=copy(lp32);cache['logits']=copy(logits);cache['target']=copy(target)
            cache['length']=ids.shape[1];cache['sample_batch']=batch
            cache['target_samples']=target_samples;cache['target_positions']=target_positions
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

