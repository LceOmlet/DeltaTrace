"""True multi-example P1 with unmodified public FA and explicit causal padding.

Each distinct example has EOS/input endpoints and its own fixed response seed.
Right-side EOS after the complete response is computed but never scored. This
requires the validated purely causal Qwen3 configuration; no left padding,
cross-example attention, prompt truncation or invented response token is used.
"""
import time
from qwen_public_fa_layer_replay import NativeLayerReplay
from qwen_signed_secant_batched_vendor_fa_gqa import propagate_signed_secant
def capture_checkpoint_batch_raw(model,ids,mask,prompt_lens,lengths):
    import torch
    assert ids.shape[0]==mask.shape[0]==len(prompt_lens)==len(lengths)
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
            for key in ['score16','score32_sum64','target_logprobs32','logits','target']:cache[key]=[]
            for sample,(plen,length) in enumerate(zip(prompt_lens,lengths)):
                assert 0<plen<length<=ids.shape[1]
                logits=result.logits[sample,plen-1:length-1];target=ids[sample,plen:length]
                lp16=logits.log_softmax(-1).gather(1,target[:,None]).flatten()
                lp32=logits.float().log_softmax(-1).gather(1,target[:,None]).flatten()
                cache['score16'].append(float(lp16.sum()))
                cache['score32_sum64'].append(float(lp32.double().sum()))
                for key,value in [('target_logprobs32',lp32),('logits',logits),('target',target)]:cache[key].append(copy(value))
            cache['prompt_len']=list(prompt_lens);cache['actual_lengths']=list(lengths);cache['length']=ids.shape[1]
            del result,logits,target,lp16,lp32
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

def capture_batch(model,requests,eos_token_id):
    """requests: before_ids[1,N], after_ids[1,N], prompt_len, one per example."""
    import torch
    assert requests and eos_token_id is not None
    lengths=[a.shape[1] for _,a,_ in requests];plens=[p for _,_,p in requests]
    batch=len(requests);n=max(lengths)
    packed=torch.full((2*batch,n),eos_token_id,device=requests[0][1].device,dtype=torch.long)
    for i,(before,after,plen) in enumerate(requests):
        assert before.shape==after.shape==(1,lengths[i]) and 0<plen<lengths[i]
        assert torch.equal(before[:,plen:],after[:,plen:])
        packed[i,:lengths[i]]=before[0];packed[batch+i,:lengths[i]]=after[0]
    master=capture_checkpoint_batch_raw(model,packed,torch.ones_like(packed),plens*2,lengths*2)
    assert master['mask'] is None and master['cos'].shape[0]==master['sin'].shape[0]==1
    endpoints=[]
    for endpoint in range(2):
        selection=slice(endpoint*batch,(endpoint+1)*batch)
        view={k:master[k] for k in ['cos','sin','mask','length']}
        view.update(prompt_len=plens,actual_lengths=lengths,endpoint_index=endpoint,paired_checkpoint=master)
        for key in ['last','norm_out']:view[key]=master[key][selection]
        for key in ['logits','target','target_logprobs32']:view[key]=torch.cat(master[key][selection],0)
        for key in ['score16','score32_sum64']:view[key]=torch.tensor(master[key][selection],dtype=torch.float64)
        endpoints.append(view)
    return tuple(endpoints)

class BatchReplay:
    def __init__(self,model,master,batch,activity):
        self.native=NativeLayerReplay(model,master,activity=activity);self.batch=batch
    def get(self,index,endpoint):
        raw=self.native[index];part=slice(endpoint*self.batch,(endpoint+1)*self.batch)
        return {key:value[part] if value is not None else None for key,value in raw.items()}
    def clear(self):self.native.clear()

class Endpoint:
    def __init__(self,paired,side):self.paired=paired;self.side=side
    def __getitem__(self,index):return self.paired.get(index,self.side)

def propagate_batch(model,before,after,pv_rule='content_P1',activity=None,finite_attention=None,finite_activity=None):
    import torch
    assert before['paired_checkpoint'] is after['paired_checkpoint']
    batch=len(before['prompt_len']);master=before['paired_checkpoint']
    paired=BatchReplay(model,master,batch,activity)
    left=dict(before);right=dict(after);left['layers']=Endpoint(paired,0);right['layers']=Endpoint(paired,1)
    try:
        result=propagate_signed_secant(model,left,right,'rescale',pv_rule=pv_rule,finite_attention=finite_attention,finite_activity=finite_activity)
        assert paired.native.calls==paired.native.auxiliary_attention_calls==len(model.model.layers)
        result.update(example_batch_size=batch,physical_endpoint_batch_size=2*batch,
            actual_lengths=before['actual_lengths'],padded_length=before['length'],prompt_lengths=before['prompt_len'],
            native_layer_replay_calls=paired.native.calls,native_layer_replay_endpoint_trajectories=2*batch*paired.native.calls,
            extra_native_fa_attention_calls=paired.native.auxiliary_attention_calls,
            extra_native_fa_attention_endpoint_trajectories=2*batch*paired.native.auxiliary_attention_calls,
            public_FA_capture_checks=paired.native.public_capture_checks,
            native_layer_boundary_checks=paired.native.boundary_checks,
            endpoint_scores32={'before':before['score32_sum64'].tolist(),'after':after['score32_sum64'].tolist()},
            padding_scope='Only trailing EOS after each complete original response, same at both endpoints. All causal work charged; no padded token loss. Original generation retained per example.',
            partial_recomputation='One actual root call and one actual replay per layer at batch2B; one additional original public FA call per replay. Content P1 generalized over independent leading sample dimension with the traceable FA finite extension; no global NxN attribution matrix or model/native backward replacement.')
        for sample,length in enumerate(before['actual_lengths']):
            assert all(v==0 for v in result['signed_full_sequence'][sample][length:])
        for audit in result['native_fa_operand_audits']:
            audit.update(extra_native_attention_calls=1,
                probability_source='No global P: finite FA tiles use actual QKV and public native LSE. Auxiliary output exactly checked, never replaces model output.',
                extra_call_count_scope='One shared batch2B physical auxiliary call per layer; both endpoint rows describe the same call.')
        return result
    finally:paired.clear()
