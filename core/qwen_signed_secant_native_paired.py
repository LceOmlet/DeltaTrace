"""Original native batch=2 root capture and decoder replay, with endpoint views.

The finite-secant arithmetic is the unchanged accepted imported function.
One physical call still performs two endpoint trajectories; count both.
"""
import time
from qwen_signed_secant_checkpointed_compiled_difference import NativeLayerReplay
from qwen_signed_secant_compiled_difference import propagate_signed_secant as full_secant_pullback

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
            result=model(input_ids=ids,attention_mask=mask,use_cache=False)
            logits=result.logits[:,prompt_len-1:-1];target=ids[:,prompt_len:]
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


class PairedReplayViews:
    def __init__(self,model,master):
        self.native=NativeLayerReplay(model,master)
        self.index=None;self.values=None;self.layout_records=[]
    def get(self,index,endpoint):
        if index!=self.index:
            raw=self.native[index]
            views=[{},{}];layouts={}
            for key,value in raw.items():
                if value is None:
                    assert key=='p';views[0][key]=views[1][key]=None
                else:
                    assert value.shape[0]==2,(key,value.shape)
                    for side in range(2):views[side][key]=value[side:side+1]
                    layouts[key]={'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),
                        'endpoint_strides':[list(v[key].stride()) for v in views],
                        'endpoint_storage_offsets':[v[key].storage_offset() for v in views]}
            self.layout_records.append({'layer':index,'tensors':layouts})
            self.values=views;self.index=index
        return self.values[endpoint]
    def clear(self):
        self.native.clear();self.values=None;self.index=None


class EndpointReplay:
    def __init__(self,paired,endpoint):self.paired=paired;self.endpoint=endpoint
    def __getitem__(self,index):return self.paired.get(index,self.endpoint)


def propagate_paired_secant(model,before,after,progress=None):
    assert before['paired_checkpoint'] is after['paired_checkpoint']
    assert before['endpoint_index']==0 and after['endpoint_index']==1
    master=before['paired_checkpoint'];paired=PairedReplayViews(model,master)
    left=dict(before);right=dict(after)
    left['layers']=EndpointReplay(paired,0);right['layers']=EndpointReplay(paired,1)
    try:
        result=full_secant_pullback(model,left,right,'rescale',progress=progress)
        assert paired.native.calls==len(model.model.layers)==36
        result['native_layer_replay_calls']=paired.native.calls
        result['native_layer_replay_endpoint_trajectories']=2*paired.native.calls
        result['native_transformer_body_replay_equivalents']=2
        result['native_layer_boundary_checks']={'paired_batch':paired.native.boundary_checks}
        result['native_paired_replay_layouts']=paired.layout_records
        result['checkpoint_retained_tensor_bytes']=master['tensor_bytes']
        result['recomputation_scope']='One original native FA batch=2 root forward and36 original native FA batch=2 decoder replays. Two endpoint trajectories in each; same full-secant arithmetic on real endpoint tensor views. Zero auxiliary FA, no surrogate/native-autograd replacement. Boundaries checked against actual batch=2 root outputs.'
        return result
    finally:paired.clear()
