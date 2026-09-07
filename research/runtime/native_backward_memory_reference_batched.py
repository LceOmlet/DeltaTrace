"""One actual native input backward for B independent fixed responses.

Uses the model's native logits_to_keep for the union of needed positions. No
parameter gradients, replacement attention, duplicate endpoint, or surrogate
loss is introduced to inflate the ordinary-backward memory reference.
"""
import time
import torch

def ordinary_batch_input_backward(model,requests,eos_token_id):
    """requests = [(actual_ids[1,N], prompt_len), ...], each a distinct sample."""
    assert requests and not model.training and all(not x.requires_grad for x in model.parameters())
    assert model.config._attn_implementation=='flash_attention_2'
    assert all(not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks for m in model.modules())
    lengths=[ids.shape[1] for ids,_ in requests];plens=[p for _,p in requests]
    assert all(ids.shape==(1,n) and 0<p<n for (ids,p),n in zip(requests,lengths))
    batch=len(requests);n=max(lengths)
    torch.cuda.empty_cache();torch.cuda.synchronize()
    resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    ids=torch.full((batch,n),eos_token_id,dtype=torch.long,device=requests[0][0].device)
    for i,(value,_) in enumerate(requests):ids[i,:lengths[i]]=value[0]
    selected_list=sorted({j for plen,length in zip(plens,lengths) for j in range(plen-1,length-1)})
    selected_lookup={position:i for i,position in enumerate(selected_list)}
    selected=torch.tensor(selected_list,device=ids.device,dtype=torch.long)
    x=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    output=model(inputs_embeds=x,attention_mask=torch.ones_like(ids),use_cache=False,output_attentions=False,logits_to_keep=selected)
    row_scores=[];token_values=[]
    for i,(plen,length) in enumerate(zip(plens,lengths)):
        start,end=selected_lookup[plen-1],selected_lookup[length-2]+1
        assert end-start==length-plen
        logits=output.logits[i,start:end]
        value=logits.float().log_softmax(-1).gather(1,ids[i,plen:length,None]).flatten()
        row_scores.append(value.double().sum());token_values.append(value.detach())
    score=torch.stack(row_scores).sum()
    gradient=torch.autograd.grad(score,x,create_graph=False,retain_graph=False)[0]
    torch.cuda.synchronize();seconds=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated()
    assert torch.isfinite(gradient).all()
    for i,length in enumerate(lengths):assert gradient[i,length:].eq(0).all()
    return {'score32_sum64':float(score.detach()),'endpoint_scores32':[float(v.detach()) for v in row_scores],
        'target_logprobs32':[v.cpu().tolist() for v in token_values],
        'peak_allocated_bytes':peak,'resident_bytes_before':resident,'incremental_peak_bytes':peak-resident,'seconds':seconds,
        'native_root_forwards':1,'native_vjps':1,'example_batch_size':batch,'physical_endpoint_batch_size':batch,
        'actual_lengths':lengths,'prompt_lengths':plens,'padded_length':n,'native_head_positions':selected.cpu().tolist(),
        'backend':model.config._attn_implementation,'parameter_gradients_enabled':False,'padding_gradient_zero':True,
        'scope':'One native forward and one actual input VJP for sum of per-example original G32 fixed-response scores. Original model/head/FA/autograd; frozen parameters, no endpoint caches, no attribution arithmetic, native selected output positions only. Trailing EOS after original complete response has no score or input gradient.'}
