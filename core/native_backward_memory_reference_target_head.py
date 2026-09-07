"""One ordinary native input VJP of the unchanged fixed-generation G32 target.

Frozen parameters, no EOS allocation/caches, no surrogate backward, no extra
diagnostic tensor retention. This is a memory ceiling, never padded by training
parameter gradients or a slower attention backend.
"""
import time
import torch


def ordinary_input_backward(model,ids,mask,prompt_len):
    selected=torch.arange(prompt_len-1,ids.shape[1]-1,device=ids.device)
    assert not model.training and all(not x.requires_grad for x in model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks for m in model.modules())
    torch.cuda.empty_cache();torch.cuda.synchronize()
    resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    x=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    output=model(inputs_embeds=x,attention_mask=mask,use_cache=False,output_attentions=False,logits_to_keep=selected)
    logits=output.logits[0]
    target=ids[0,prompt_len:]
    score=logits.float().log_softmax(-1).gather(1,target[:,None]).double().sum()
    gradient=torch.autograd.grad(score,x,create_graph=False,retain_graph=False)[0]
    torch.cuda.synchronize();seconds=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated()
    assert torch.isfinite(gradient).all()
    result={'score32_sum64':float(score.detach()),'peak_allocated_bytes':peak,'resident_bytes_before':resident,
        'incremental_peak_bytes':peak-resident,'seconds':seconds,'native_root_forwards':1,'native_vjps':1,
        'backend':model.config._attn_implementation,'parameter_gradients_enabled':False,
        'scope':'Ordinary one native forward and one input-gradient backward of actual G32, same model/input/dtype/backend. No baseline caches, attribution arithmetic, extra loss, or parameter-gradient inflation.'}
    return result
