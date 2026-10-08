"""Research finite-rule endpoints, composed from the pinned public FA owner.

This module performs no model inference. It requests the original coefficient
suffix on the factual strict-past K/V history, then adds the own reference key
and value. The existing compiled boundary owner supplies scalar contractions.
"""
import torch
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis


def prepare_conditional_endpoints(captures,layout,scale,boundaries):
    q0,q1=captures['query'][0::2],captures['query'][1::2]
    k0,k1=captures['key'][0::2],captures['key'][1::2]
    v0,v1=captures['value'][0::2],captures['value'][1::2]
    b,h,t,d=q0.shape
    starts=layout._query_starts
    if starts is None:
        starts=torch.full((b,),layout.query_start,device=q0.device,dtype=torch.int32)
    lengths=layout._tensor
    cuts=layout._starts if layout._starts is not None else starts
    own0,own1=boundaries.attention_own_scores(q0,q1,k0,starts,scale)
    # Exact owner lengths determine the two representations. Causality itself
    # is implemented solely by the public FA bottom-right causal mask.
    positions=starts[:,None]+torch.arange(t,device=q0.device)[None,:]
    query_mask=(positions>=cuts[:,None]) & (positions<lengths[:,None]) & (positions>0)
    key_mask=torch.arange(layout.padded_length,device=q0.device)[None,:]<(lengths[:,None]-1)
    packed_q,indices_q,cuq,maxq=unpad_input(q0.transpose(1,2),query_mask)
    packed_k,indices_k,cuk,maxk=unpad_input(k1.transpose(1,2),key_mask)
    packed_v=index_first_axis(v1.transpose(1,2).reshape(b*layout.padded_length,-1,d),indices_k)
    if maxq:
        past,lse,unused=flash_attn_varlen_func(packed_q,packed_k,packed_v,cuq,cuk,maxq,maxk,
            dropout_p=0.,softmax_scale=scale,causal=True,return_attn_probs=True)
        if unused is not None and unused.numel():
            raise ValueError('Unexpected quadratic probability output.')
        if lse.ndim!=2 or lse.shape!=(h,packed_q.shape[0]):
            raise ValueError('Installed public varlen FA LSE representation changed.')
        past=pad_input(past,indices_q,b,t)
        lse=pad_input(lse.transpose(0,1),indices_q,b,t).transpose(1,2)
    else:
        past=q0.new_zeros((b,t,h,d))
        lse=own0.new_zeros((b,h,t))
    # Omitted rows have an empty strict-past sum, including logical row zero.
    # Derive that representation from exact positions, not an FA score value.
    lse=torch.where(query_mask[:,None,:],lse,torch.full_like(lse,-torch.inf))
    attention0,lse0=boundaries.attention_own_endpoint(past,lse,own0,v0,starts)
    return dict(attention0=attention0,lse0=lse0,own_q0k0=own0,own_q1k0=own1,query_starts=starts)
