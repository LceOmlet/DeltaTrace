"""Observe equal causal prefixes in the original paired Qwen forward.

No model, cache, DT or operator is replaced. This records prefix invariance
and the first small native projection mismatch; it sets no acceptance tolerance.
"""
import torch


def inspect(runtime, selected, reference, cases, labels, producer):
    from qwen35_answer_finite import selected_target_log_probs

    pair=torch.stack((reference,selected),1).flatten(0,1)
    length=pair.shape[1]
    changed=reference!=selected
    cut=int(torch.where(changed,torch.arange(length,device=pair.device),length).amin())
    assert torch.equal(pair[0::2,:cut],pair[1::2,:cut])
    selection=producer.packed_answer_targets(cases,[[0]]*4,length,pair.device,outcome_token_ids=labels)
    positions=selection.positions.unique(sorted=True)
    records=[];first={};handles=[]

    def observe(name,small_projection=False):
        def hook(module,args,out):
            value=out[0] if isinstance(out,tuple) else out
            a,b=value[0::2,:cut],value[1::2,:cut]
            delta=a.float()-b.float()
            row=dict(name=name,equal=torch.equal(a,b),max_abs=float(delta.abs().max()),
                     relative_l2=float(delta.norm()/b.float().norm().clamp_min(1e-30)))
            if small_projection and not row['equal'] and not first:
                i,t,c=(a!=b).nonzero()[0].tolist()
                x0=args[0][2*i,t].detach().cpu().double()
                x1=args[0][2*i+1,t].detach().cpu().double()
                weight=module.weight
                if hasattr(weight,'full_tensor'):weight=weight.full_tensor()
                weight=weight[c].detach().cpu().double()
                first.update(name=name,pair=i,position=t,output_channel=c,
                             input_equal=torch.equal(x0,x1),
                             native=[float(a[i,t,c]),float(b[i,t,c])],
                             cpu_float64_dot=[float(x0@weight),float(x1@weight)])
            records.append(row)
        return hook

    for i,layer in enumerate(runtime.model.model.language_model.layers):
        handles.append(layer.register_forward_hook(observe(f'layer.{i}')))
        if layer.block_type=='linear_attention':
            for name in ('in_proj_a','in_proj_b'):
                module=getattr(layer.linear_attn,name)
                leaf=getattr(module,'base_layer',module)
                handles.append(leaf.register_forward_hook(observe(f'layer.{i}.{name}',True)))
    try:
        with torch.no_grad():
            output=runtime.model.forward_root(input_ids=pair,attention_mask=torch.ones_like(pair),
                                             use_cache=False,logits_to_keep=positions)
            lp=selected_target_log_probs(output.logits[selection.paired_samples,
                torch.searchsorted(positions,selection.paired_positions)],selection).cpu()
    finally:
        for handle in handles:handle.remove()
    return dict(scope=__doc__,prefix_tokens=cut,length=length,observations=records,
                first_unequal_small_projection=first,log_probs=lp.tolist())
