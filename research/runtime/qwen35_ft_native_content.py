"""Explicit corrected FT content decomposition using actual FA/FLA adjoints.

Not an unchanged official FT implementation and not DeltaTrace finite propagation.
Freeze actual endpoint routes and output gains; use native public value gradients
to avoid identity-valued/T-square probes. No model forward/backward is replaced.
"""
import torch
from finite_fla_gpu import _mm


class RightPaddedBatch:
    def __init__(self,lengths,padded_length,device):
        if not lengths or any(not isinstance(n,int) or n<1 or n>padded_length for n in lengths):
            raise ValueError('Invalid true sequence lengths.')
        self.lengths=tuple(lengths);self.width=padded_length;self.batch=len(lengths)
        self.mask=torch.arange(padded_length,device=device)[None,:]<torch.tensor(lengths,device=device)[:,None]
        self.indices=self.mask.flatten().nonzero().flatten()
        self.cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device=device,dtype=torch.int32)
        self.maximum=max(lengths)

    def pack(self,x):
        assert x.shape[:2]==(self.batch,self.width)
        return x.flatten(0,1)[self.indices].contiguous()

    def unpack(self,x):
        from flash_attn.bert_padding import pad_input
        return pad_input(x,self.indices,self.batch,self.width)


def native_value_gradient(output,value,seed,events=None):
    """Call genuine autograd and optionally observe native node execution threads."""
    handles=[];todo=[output.grad_fn];seen=set()
    while todo:
        node=todo.pop()
        if node is None or node in seen:continue
        seen.add(node);name=type(node).__name__
        if events is not None and any(x in name for x in ('FlashAttn','GatedDelta','CausalConv')):
            def observed(inputs,outputs,name=name):
                events.append({'node':name,'input_gradients_present':[x is not None for x in inputs]})
            handles.append(node.register_hook(observed))
        todo.extend(x for x,_ in node.next_functions if x is not None)
    try:return torch.autograd.grad(output,value,seed.to(output.dtype),retain_graph=False,create_graph=False)[0].detach()
    finally:
        for h in handles:h.remove()


def fa_content_components(module,capture,sink_weights,layout,timed,events=None):
    """Original endpoint B rows only; one public FA forward/backward, true varlen.

    Expand compact KV into query heads only in the explicit auxiliary call, so
    native GQA reduction cannot destroy separate head/O-projection information.
    Public FA kernels remain the installed implementation; discarded dQ/dK paid.
    """
    from flash_attn import flash_attn_varlen_func
    c=capture;b,t=sink_weights.shape;h=module.config.num_attention_heads;d=module.head_dim
    assert c['query'].shape==(b,h,t,d) and module.attention_dropout==0
    groups=module.num_key_value_groups
    q=layout.pack(c['query'].transpose(1,2))
    k=layout.pack(c['key'].repeat_interleave(groups,dim=1).transpose(1,2))
    values=c['value'].repeat_interleave(groups,dim=1).transpose(1,2).contiguous()
    v=layout.pack(values).detach().requires_grad_(True)
    gate=c['q_proj_output'].view(b,t,h,2*d)[...,d:]
    gain=gate.sigmoid().float();seed=sink_weights[:,:,None,None]*gain
    with torch.enable_grad():
        out=timed('public_FA_expanded_heads_forward',lambda:flash_attn_varlen_func(q,k,v,layout.cu,layout.cu,
            layout.maximum,layout.maximum,dropout_p=0.0,softmax_scale=module.scaling,causal=True))
        dv=timed('public_FA_value_backward',lambda:native_value_gradient(out,v,layout.pack(seed),events))
    dv=layout.unpack(dv);components=values.float()*dv.float()
    return components,{'auxiliary_core':layout.unpack(out.detach()),'gain':gain,'seed_native':seed.to(values.dtype),
        'value_gradient':dv,'value':values,'query':c['query'],'key':c['key'],'native_core':c['attention_output']}


def gdn_content_components(module,capture,endpoints,sink_weights,scale,timed,events=None):
    """Correct output norm/gate and return sources BEFORE the causal convolution.

    Q/K/g/beta and output gains are frozen real endpoint values. The convolution
    uses its native activation=None transpose and a bounded frozen SiLU gain,
    sigmoid(preactivation). Fused-SiLU/linear-conv rounding remains measurable.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    c=capture;e=endpoints;b,t=sink_weights.shape;h=module.num_v_heads;d=module.head_v_dim
    assert module.conv1d.bias is None and module.norm.bias is None and module.norm.activation=='silu'
    core=e['o'].float();gain=(torch.rsqrt(core.square().mean(-1,keepdim=True)+module.norm.eps)
        *module.norm.weight.float()*torch.nn.functional.silu(c['z'].float()))
    seed=sink_weights[:,:,None,None]*gain
    v=e['v'].detach().requires_grad_(True)
    with torch.enable_grad():
        # q/k already are the actual normalized inputs of the native chunk
        # forward. Do not normalize them a second time in this auxiliary call.
        out,last=timed('public_FLA_normalized_forward',lambda:chunk_gated_delta_rule(e['q'],e['k'],v,
            g=e['raw_g'],beta=e['beta'],scale=scale,initial_state=None,output_final_state=False,
            use_qk_l2norm_in_kernel=False))
        assert last is None
        dv=timed('public_FLA_value_backward',lambda:native_value_gradient(out,v,seed,events))
        projected=c['projected_qkv'].detach().requires_grad_(True)
        pre=timed('public_linear_conv_forward',lambda:module.causal_conv1d_fn(projected,module.conv1d.weight.squeeze(1),activation=None))
        offset=2*module.key_dim
        pv=pre[:,offset:].transpose(1,2).reshape(b,t,h,d)
        frozen_silu_gain=pv.float().sigmoid()
        pre_seed=torch.zeros_like(pre)
        pre_seed[:,offset:]=(dv.float()*frozen_silu_gain).flatten(2).transpose(1,2).to(pre.dtype)
        dprojected=timed('public_linear_conv_backward',lambda:native_value_gradient(pre,projected,pre_seed,events))
    source=c['projected_qkv'][:,offset:].transpose(1,2).reshape(b,t,h,d)
    source_gradient=dprojected[:,offset:].transpose(1,2).reshape(b,t,h,d)
    components=source.float()*source_gradient.float()
    return components,{'auxiliary_core':out.detach(),'native_core':e['o'],'gain':gain,'seed_native':seed.to(v.dtype),
        'value_gradient':dv,'post_conv_components':e['v'].float()*dv.float(),'source':source,'source_gradient':source_gradient,
        'pre_value':pv.detach(),'frozen_silu_gain':frozen_silu_gain,'pre_value_seed':pre_seed[:,offset:].transpose(1,2).reshape(b,t,h,d),
        'value':e['v'],'conv_weights':module.conv1d.weight.squeeze(1)[offset:].view(h,d,-1),
        'q':e['q'],'k':e['k'],'raw_g':e['raw_g'],'beta':e['beta']}


def project_components(components,out_weight):
    """Vendor BF16 GEMM, FP32 accumulation, output rounded to native FT dtype."""
    b,j,h,d=components.shape;c=out_weight.shape[0]
    a=components.permute(2,0,1,3).reshape(h,b*j,d)
    weight=out_weight.view(c,h,d).permute(1,2,0)
    return _mm(a,weight).to(out_weight.dtype).reshape(h,b,j,c).permute(1,2,0,3)


def aggregate_native_content_ft(components,out_weight,decoder_capture,sink_weights,proximity,chunk_tokens=32):
    """FT proximity/normalization with corrected native-structure contributions.

    This initially uses the actual author's proximity function and existing
    vendor matrix operations. Source positions are streamed32 at a time; no
    [B,T,H,C] contribution tensor is retained. All examples are truly batched.
    """
    b,t,h,d=components.shape;native_dtype=out_weight.dtype;c=decoder_capture
    assert sink_weights.shape==(b,t) and chunk_tokens>0
    mid=(c['post_norm_input'].float()*sink_weights[:,:,None]).sum(1)
    residual=(c['input_norm_input'].float()*sink_weights[:,:,None]).sum(1)
    residual_proximity=proximity(residual,mid)
    numer=torch.empty((b,t),dtype=native_dtype,device=components.device)
    head_numer=torch.zeros((b,h),dtype=native_dtype,device=components.device)
    reconstructed=torch.zeros((b,h,out_weight.shape[0]),dtype=torch.float64,device=components.device)
    for start in range(0,t,chunk_tokens):
        end=min(t,start+chunk_tokens);contribution=project_components(components[:,start:end],out_weight)
        prox=proximity(contribution.float(),mid[:,None,None,:])
        numer[:,start:end]=prox.sum(2).to(native_dtype)
        head_numer+=prox.sum(1).to(native_dtype)
        reconstructed+=contribution.double().sum(1)
    denominator=numer.float().sum(1)+residual_proximity+1e-12
    return {'token_scores':numer.float()/denominator[:,None],'head_scores':head_numer.float()/denominator[:,None],
        'residual_score':residual_proximity/denominator,'numerator':numer,'head_numerator':head_numer,
        'residual_proximity':residual_proximity,'mid_sum':mid,'projected_head_sums':reconstructed}
