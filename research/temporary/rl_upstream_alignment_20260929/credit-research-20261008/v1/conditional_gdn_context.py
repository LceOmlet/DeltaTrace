"""Research-only GDN conditional memory seam; original owners still execute.

No full model/source replay, copied recurrence, credit correction or new Q/V.
All sources are processed, using original head groups and native chunk tiles.
The current incoming cotangent/norm-gate rule remains the existing DT rule.
"""
import torch
import torch.nn.functional as F

from conditional_conv_windows import conditional_conv_windows, native_qkv
from tiled_conditional_memory import conditional_memory_tiles
from finite_fla_gpu import native_input_adjoints, mixed_coefficients, exp_secant_factors
from qwen35_gdn_finite import _l2_pullback, _scalar_secant, _silu_derivative


def conditional_gdn_context(module, c, e, do, scale, *, cut, seed_exponent,
                            restore_seed_range, head_batch, release_consumed,
                            tile_size=4096, observer=None):
    B,T,H,V=do.shape
    K=module.head_k_dim
    repeat=module.num_v_heads//module.num_k_heads
    head_batch=H if head_batch is None else head_batch
    assert head_batch % repeat == 0 and module.conv1d.weight.shape[-1] == 4
    dtype=c['projected_qkv'].dtype
    mqkv=torch.zeros((B,T,module.conv_dim),device=do.device,dtype=dtype)
    mb=torch.zeros((B,T,H),device=do.device,dtype=torch.float32)
    ma=torch.zeros_like(mb)
    context=c['projected_qkv'].shape[-1]-T
    N=(T-cut+63)//64

    def transfer(value):return value.to(device=do.device).contiguous()
    def ahead(value,lag,start,stop):
        part=value[:,start+lag:min(stop+lag,value.shape[1])]
        return F.pad(part,(0,0)*(part.ndim-2)+(0,stop-start-part.shape[1]))

    for head in range(0,H,head_batch):
        count=min(head_batch,H-head)
        indices=torch.cat((torch.arange(head//repeat*K,(head+count)//repeat*K),
            torch.arange(module.key_dim+head//repeat*K,module.key_dim+(head+count)//repeat*K),
            torch.arange(2*module.key_dim+head*V,2*module.key_dim+(head+count)*V)))
        group={name:transfer(value[:,cut//64 if name=='h' else cut:,head:head+count])
               for name,value in e.items() if name!='o'}
        factual={name:value[1::2].contiguous() for name,value in group.items()}
        coincident={name:value.repeat_interleave(2,0) for name,value in factual.items()}
        adjoints=native_input_adjoints(coincident,do[:,cut:,head:head+count].contiguous(),scale)
        base,detail=mixed_coefficients(coincident,adjoints,scale,diagnostics=True)
        def unpack(value):return value.reshape(B,count,N*64,K)[:,:,:T-cut].permute(0,2,1,3).contiguous()
        L,r0=unpack(detail['L']),unpack(detail['r0'])
        del coincident,detail
        # Channel groups are independent in the original depthwise operator.
        # Preserve the real convolution history before the native FLA cut.
        local_indices=indices.to(c['projected_qkv'].device)
        projected=transfer(c['projected_qkv'].index_select(1,local_indices)).transpose(1,2).contiguous().transpose(1,2)
        x,x0=projected[1::2,:,context+cut:],projected[0::2,:,context+cut:]
        if context+cut:
            history=projected[1::2,:,max(0,context+cut-3):context+cut]
            history=F.pad(history,(3-history.shape[-1],0))
        elif c.get('conv_initial_states') is not None:
            history=transfer(c['conv_initial_states'][1::2].index_select(1,indices.to(c['conv_initial_states'].device)))
        else:history=x.new_zeros(B,3,x.shape[1]).transpose(1,2)
        history=history.transpose(1,2).contiguous().transpose(1,2)
        weight=module.conv1d.weight.squeeze(1).index_select(0,indices.to(module.conv1d.weight.device))
        pre=module.causal_conv1d_fn(x,weight,initial_states=history,activation=None).transpose(1,2)
        fused=transfer(c['conv_output'][1::2,:,cut:].index_select(1,indices.to(c['conv_output'].device))).transpose(1,2)
        # The native wrapper's actual raw q dtype fixes the cast boundary.
        raw_dtype=c['raw_q'].dtype
        normalized,raw=native_qkv(fused,key_heads=count//repeat,value_heads=count,
            key_dim=K,value_dim=V,operand_dtype=raw_dtype,return_raw=True)
        for name in ('q','k','v'):
            if not torch.equal(normalized[name].to(factual[name].dtype),factual[name]):
                raise ValueError('Conditional GDN raw/normalized capture interface differs: '+name)
        del normalized
        held={}
        def conditional(start,stop):
            pack=conditional_conv_windows(module.causal_conv1d_fn,x,x0,weight,
                initial=history,start=start,stop=stop,retain_pre_graph=True)
            norm,raw0=native_qkv(pack['output'],key_heads=count//repeat,value_heads=count,
                key_dim=K,value_dim=V,operand_dtype=raw_dtype,return_raw=True)
            held.update(pack=pack,raw=raw0)
            return {name:value.to(factual[name].dtype) for name,value in norm.items()}
        exponent=None if seed_exponent is None else seed_exponent[:,:,head:head+count]
        alpha0=group['raw_g'][0::2].float().exp()
        beta0=group['beta'][0::2]
        for start,stop,coefficient,calls in conditional_memory_tiles(factual,base,L,r0,
                adjoints,conditional,alpha0,beta0,scale,tile_size=tile_size):
            pack,raw0=held.pop('pack'),held.pop('raw')
            if exponent is not None:
                parts={name+str(lag):coefficient[name][lag] for name in ('q','k','v') for lag in range(4)}
                parts.update(alpha=coefficient['alpha'],beta=coefficient['beta'])
                parts=restore_seed_range(parts,exponent)
                coefficient={name:torch.stack([parts[name+str(lag)] for lag in range(4)])
                             for name in ('q','k','v')}|{name:parts[name] for name in ('alpha','beta')}
                del parts
            fact_raw={name:torch.stack([ahead(raw[name],lag,start,stop) for lag in range(4)]) for name in ('q','k')}
            q=_l2_pullback(raw0['q'],fact_raw['q'],coefficient['q'])
            k=_l2_pullback(raw0['k'],fact_raw['k'],coefficient['k'])
            q=q.reshape(4,B,stop-start,count//repeat,repeat,K).sum(-2)
            k=k.reshape(4,B,stop-start,count//repeat,repeat,K).sum(-2)
            mconv=torch.cat((q.flatten(-2),k.flatten(-2),coefficient['v'].flatten(-2)),dim=-1)
            pf=torch.stack([ahead(pre,lag,start,stop) for lag in range(4)]).float()
            yf=torch.stack([ahead(fused,lag,start,stop) for lag in range(4)]).float()
            pc,yc=pack['pre'].float(),pack['output'].float()
            mpre=mconv*_scalar_secant(pc,pf,yc,yf,_silu_derivative(pc))
            mpre=mpre*pack['valid'][:,None,:,None]
            seed=mpre.permute(1,2,3,0).reshape(B*(stop-start),len(indices),4).to(pack['native_pre'].dtype)
            with torch.enable_grad():
                dx,=torch.autograd.grad(pack['native_pre'],pack['native_input'],seed)
            grad=dx[:,:,0].reshape(B,stop-start,len(indices)).to(mqkv.dtype)
            mqkv[:,cut+start:cut+stop,indices.to(mqkv.device)]=grad
            b0=transfer(c['b'][0::2,cut+start:cut+stop,head:head+count]).float()
            b1=transfer(c['b'][1::2,cut+start:cut+stop,head:head+count]).float()
            sigmoid=b0.sigmoid()
            mb[:,cut+start:cut+stop,head:head+count]=coefficient['beta']*_scalar_secant(b0,b1,
                group['beta'][0::2,start:stop].float(),group['beta'][1::2,start:stop].float(),sigmoid*(1-sigmoid))
            a0=transfer(c['a'][0::2,cut+start:cut+stop,head:head+count]).float()
            a1=transfer(c['a'][1::2,cut+start:cut+stop,head:head+count]).float()
            g0=group['raw_g'][0::2,start:stop].float();g1=group['raw_g'][1::2,start:stop].float()
            ga0=-module.A_log[head:head+count].float().exp()*(a0+module.dt_bias[head:head+count].float()).sigmoid()
            exponential,ratio=exp_secant_factors(g0,g1)
            ma[:,cut+start:cut+stop,head:head+count]=coefficient['alpha']*exponential*ratio*_scalar_secant(a0,a1,g0,g1,ga0)
            if observer is not None:observer(head,start,stop,calls)
            del pack,raw0,coefficient,fact_raw,q,k,mconv,pf,yf,pc,yc,mpre,seed,dx,grad,b0,b1,a0,a1,g0,g1,ga0,sigmoid,exponential,ratio
        del group,factual,adjoints,base,L,r0,projected,x,x0,history,weight,pre,fused,raw
    if release_consumed:
        e.clear()
        for name in ('raw_q','raw_k','a','b','projected_qkv','conv_output','conv_initial_states'):
            c.pop(name,None)
    return mqkv,mb,ma
