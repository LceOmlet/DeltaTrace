from compiled_fp64_difference_audit import audit_difference, compiled_fp64_difference_dot
from compiled_native_probability_audit import compiled_probability, compiled_pv_audit
from native_half_linear import native_half_linear
from compiled_fp64_audit import compiled_fp64_dot
"""Original strong secant with deferred audit scalar transfer. No model or attribution-rule replacement."""
"""Independent Qwen3 finite-secant attribution from two native forward caches.

No FT scores or surrogate model. Manual attribution pullbacks are explicitly
different from native derivatives. The rc_gate variant splits signed incoming
gate-projection contributions before SiLU; it does not claim to solve every
interaction or to equal individual-token deletion effects.
"""
import math
import time
from signed_secant_rules import nonlinear_reveal_cancel
from compiled_logprob_seed import logprob_secant_seed
from compiled_finite_rules import softmax_secant_pullback, rmsnorm_secant_pullback


def propagate_signed_secant(model,before,after,variant='rescale',progress=None):
    import torch
    assert variant in ['rescale','rc_gate']
    assert before['length']==after['length'] and before['prompt_len']==after['prompt_len']
    assert torch.equal(before['target'],after['target'])
    assert torch.equal(before['cos'],after['cos']) and torch.equal(before['sin'],after['sin'])
    assert before['mask'] is None and after['mask'] is None
    device=next(model.parameters()).device
    native_dtype=next(model.parameters()).dtype
    assert native_dtype==torch.float16
    torch.cuda.synchronize();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    ledger=[];checks=[];rc=[];pending_ledger=[];fa_operand_audits=[]
    def f(x):
        return x.to(device=device,dtype=torch.float32)
    def dot(a,b):
        if isinstance(b,tuple):
            assert a.dtype==torch.float32
            return compiled_fp64_difference_dot(a,b[0],b[1])
        return compiled_fp64_dot(a,b)
    def record(name,m,delta,inputs):
        output=dot(m,delta)
        incoming=[dot(mm,dd) for mm,dd in inputs]
        pending_ledger.append((name,output,incoming))
    def norm_back(module,x0,x1,y0,y1,m,name):
        answer=rmsnorm_secant_pullback(x0,x1,f(module.weight),m,module.variance_epsilon)
        record(name,m,audit_difference(y1,y0),[(answer,audit_difference(x1,x0))])
        return answer
    def linear_back(module,x0,x1,y0,y1,m,name):
        answer=native_half_linear(m,module.weight)
        record(name,m,audit_difference(y1,y0),[(answer,audit_difference(x1,x0))])
        return answer
    def rotate_half(x):
        first,second=x.chunk(2,dim=-1)
        return torch.cat((-second,first),dim=-1)
    def rotation_transpose(m,cos,sin):
        # R(x)=x*cos+J(x)*sin; J^T=-J. Do not assume sin halves equal.
        return m*cos-rotate_half(m*sin)
    def native_attention_operands(values,layer):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        qs,ks,vs=values['fa_q'],values['fa_k'],values['fa_v']
        n=before['length'];h=layer.self_attn.head_dim;groups=layer.self_attn.num_key_value_groups
        reconstructed_q,reconstructed_k=apply_rotary_pos_emb(values['qnorm'].transpose(1,2),values['knorm'].transpose(1,2),before['cos'],before['sin'])
        assert torch.equal(reconstructed_q,qs.transpose(1,2)) and torch.equal(reconstructed_k,ks.transpose(1,2))
        assert torch.equal(values['v'].view(1,n,-1,h),vs)
        assert torch.equal(values['fa_out'].reshape_as(values['concat']),values['concat'])
        # These are actual native Q/K/V/LSE; no auxiliary attention call.
        # Probabilities below are explicitly reconstructed attribution operands,
        # never returned to the model or represented as a native P output.
        lse=values['fa_lse'].float()
        q,k,v=qs.transpose(1,2).float(),ks.transpose(1,2).float(),vs.transpose(1,2).float()
        kr=k.repeat_interleave(groups,dim=1);vr=v.repeat_interleave(groups,dim=1)
        # These are explicit attribution scores computed from actual native Q/K,
        # not hidden native logits or replacement model attention outputs.
        raw=(q@kr.transpose(-1,-2))*layer.self_attn.scaling
        assert lse.shape==q.shape[:3] and lse.dtype==torch.float32
        z,prob,audit=compiled_probability(raw,lse)
        valid,probability_relative,row_error=audit.cpu().tolist()
        assert valid==1.,'Native-LSE attribution probability validity failed'
        pv=(prob@vr).transpose(1,2).reshape_as(values['concat'])
        actual=values['concat'].float()
        pv_relative,pv_maximum=compiled_pv_audit(pv,actual).cpu().tolist()
        fa_operand_audits.append({'layer':layer.self_attn.layer_idx,'endpoint':'before' if len(fa_operand_audits)%2==0 else 'after',
            'actual_native_qkv_and_output_exact':True,'auxiliary_dropout':None,'model_dropout':0.,'auxiliary_output_used_by_model':False,'extra_native_attention_calls':0,
            'probability_source':'Explicit FP32 exp(QK*scale minus actual native FA LSE), causal masked. Attribution operand only, not a native returned matrix or model output.',
            'probability_relative_l2_to_explicit_QK':probability_relative,
            'probability_row_sum_max_error':row_error,
            'PV_relative_l2_to_actual_FA':pv_relative,
            'PV_max_error_to_actual_FA':pv_maximum})
        assert fa_operand_audits[-1]['probability_relative_l2_to_explicit_QK']<=1e-3
        assert fa_operand_audits[-1]['probability_row_sum_max_error']<=1e-3
        assert fa_operand_audits[-1]['PV_relative_l2_to_actual_FA']<=1e-3
        return q,k,v,kr,vr,prob,raw,z
    with torch.no_grad():
        z0=f(before['logits']);z1=f(after['logits'])
        seed=logprob_secant_seed(z0,z1,after['target'].to(device))
        g_delta=after['score32_sum64']-before['score32_sum64']
        seed_credit=dot(seed,audit_difference(z1,z0))
        pending_ledger.append(('target_logprob',g_delta,[seed_credit]))
        start=before['prompt_len']-1
        m_final=torch.zeros_like(f(before['norm_out']))
        m_final[0,start:-1]=native_half_linear(seed,model.lm_head.weight)
        record('lm_head',seed,audit_difference(z1,z0),[(m_final,audit_difference(f(after['norm_out']),f(before['norm_out'])))])
        del seed,z0,z1
        m=norm_back(model.model.norm,f(before['last']),f(after['last']),f(before['norm_out']),f(after['norm_out']),m_final,'final_norm')
        del m_final
        for li in reversed(range(len(model.model.layers))):
            layer=model.model.layers[li]
            raw0=before['layers'][li];raw1=after['layers'][li]
            # Attention probabilities stay on CPU until their dedicated stage.
            a={k:f(v) for k,v in raw0.items() if k!='p' and not k.startswith('fa_')}
            b={k:f(v) for k,v in raw1.items() if k!='p' and not k.startswith('fa_')}
            name=lambda part:f'layer{li}.{part}'
            record(name('mlp_residual_add'),m,audit_difference(b['out'],a['out']),[(m,audit_difference(b['mid'],a['mid'])),(m,audit_difference(b['mlp_out'],a['mlp_out']))])
            m_mid=m
            m_product=linear_back(layer.mlp.down_proj,a['product'],b['product'],a['mlp_out'],b['mlp_out'],m,name('down_proj'))
            # Match the actual native SiLU/product values; no model replay.
            silu0=torch.nn.functional.silu(raw0['gate'].to(device)).float()
            silu1=torch.nn.functional.silu(raw1['gate'].to(device)).float()
            for raw,silu in [(raw0,silu0),(raw1,silu1)]:
                product=silu.to(native_dtype)*raw['up'].to(device)
                assert torch.equal(product,raw['product'].to(device)),'Native SwiGLU product mismatch'
            mg=m_product*(a['up']+b['up'])*.5
            mu=m_product*(silu0+silu1)*.5
            record(name('swiglu_product'),m_product,audit_difference(b['product'],a['product']),[(mg,audit_difference(silu1,silu0)),(mu,audit_difference(b['up'],a['up']))])
            m_x_up=linear_back(layer.mlp.up_proj,a['mlp_in'],b['mlp_in'],a['up'],b['up'],mu,name('up_proj'))
            if variant=='rescale':
                difference=b['gate']-a['gate']
                sigmoid=torch.sigmoid(a['gate'])
                derivative=sigmoid*(1+a['gate']*(1-sigmoid))
                multiplier=torch.where(difference!=0,(silu1-silu0)/torch.where(difference!=0,difference,torch.ones_like(difference)),derivative)
                m_gate=mg*multiplier
                record(name('silu_rescale'),mg,audit_difference(silu1,silu0),[(m_gate,difference)])
                m_x_gate=linear_back(layer.mlp.gate_proj,a['mlp_in'],b['mlp_in'],a['gate'],b['gate'],m_gate,name('gate_proj'))
                del difference,sigmoid,derivative,multiplier,m_gate
            else:
                dx=b['mlp_in']-a['mlp_in']
                weight=f(layer.mlp.gate_proj.weight)
                wp=weight.clamp_min(0);wn=weight.clamp_max(0)
                positive=dx.clamp_min(0)@wp.T+dx.clamp_max(0)@wn.T
                negative=dx.clamp_min(0)@wn.T+dx.clamp_max(0)@wp.T
                cp,cn=nonlinear_reveal_cancel(torch.nn.functional.silu,a['gate'],positive,negative)
                sigmoid=torch.sigmoid(a['gate']);derivative=sigmoid*(1+a['gate']*(1-sigmoid))
                mp=torch.where(positive>0,cp/torch.where(positive>0,positive,torch.ones_like(positive)),derivative)
                mn=torch.where(negative<0,cn/torch.where(negative<0,negative,torch.ones_like(negative)),derivative)
                mp=mg*mp;mn=mg*mn
                forward_sign=mp@wp+mn@wn
                reverse_sign=mp@wn+mn@wp
                m_x_gate=torch.where(dx>0,forward_sign,torch.where(dx<0,reverse_sign,(forward_sign+reverse_sign)*.5))
                record(name('gate_proj_silu_reveal_cancel'),mg,audit_difference(silu1,silu0),[(m_x_gate,dx)])
                rc.append({'layer':li,'incoming_positive_total':float(positive.double().sum()),
                    'incoming_negative_total':float(negative.double().sum()),'positive_channel_output_total':float(cp.double().sum()),
                    'negative_channel_output_total':float(cn.double().sum()),
                    'max_gate_endpoint_rounding_residual':float((a['gate']+positive+negative-b['gate']).abs().max()),
                    'both_nonzero_channels':int(((positive>0)&(negative<0)).sum()),
                    'zero_native_gate_delta_with_both_channels':int(((b['gate']==a['gate'])&(positive>0)&(negative<0)).sum())})
                del dx,weight,wp,wn,positive,negative,cp,cn,sigmoid,derivative,mp,mn,forward_sign,reverse_sign
            m_mlp_input=m_x_gate+m_x_up
            m_norm=norm_back(layer.post_attention_layernorm,a['mid'],b['mid'],a['mlp_in'],b['mlp_in'],m_mlp_input,name('post_attention_norm'))
            m_mid=m_mid+m_norm
            record(name('attention_residual_add'),m_mid,audit_difference(b['mid'],a['mid']),[(m_mid,audit_difference(b['x'],a['x'])),(m_mid,audit_difference(b['attn_out'],a['attn_out']))])
            m_x_residual=m_mid
            m_concat=linear_back(layer.self_attn.o_proj,a['concat'],b['concat'],a['attn_out'],b['attn_out'],m_mid,name('o_proj'))
            q0,k0,v0,kr0,vr0,p0,score0,z0=native_attention_operands(raw0,layer)
            q1,k1,v1,kr1,vr1,p1,score1,z1=native_attention_operands(raw1,layer)
            n=before['length'];heads=q0.shape[1];h=layer.self_attn.head_dim;groups=layer.self_attn.num_key_value_groups
            mout=m_concat.view(1,n,heads,h).transpose(1,2)
            mp=mout@((vr0+vr1)*.5).transpose(-1,-2)
            mvr=((p0+p1)*.5).transpose(-1,-2)@mout
            record(name('attention_pv'),mout,(b['concat']-a['concat']).view(1,n,heads,h).transpose(1,2),[(mp,audit_difference(p1,p0)),(mvr,audit_difference(vr1,vr0))])
            mz=softmax_secant_pullback(z0,z1,mp)
            record(name('attention_softmax'),mp,audit_difference(p1,p0),[(mz,audit_difference(score1,score0))])
            scaling=layer.self_attn.scaling
            mq=(mz@((kr0+kr1)*.5))*scaling
            mkr=(mz.transpose(-1,-2)@((q0+q1)*.5))*scaling
            record(name('attention_qk_scaled'),mz,audit_difference(score1,score0),[(mq,audit_difference(q1,q0)),(mkr,audit_difference(kr1,kr0))])
            mk=mkr.reshape(1,k0.shape[1],groups,n,h).sum(2)
            mv=mvr.reshape(1,v0.shape[1],groups,n,h).sum(2)
            cos=f(before['cos']).unsqueeze(1);sin=f(before['sin']).unsqueeze(1)
            mqn=rotation_transpose(mq,cos,sin).transpose(1,2)
            mkn=rotation_transpose(mk,cos,sin).transpose(1,2)
            record(name('q_rope'),mq,audit_difference(q1,q0),[(mqn,audit_difference(b['qnorm'],a['qnorm']))])
            record(name('k_rope'),mk,audit_difference(k1,k0),[(mkn,audit_difference(b['knorm'],a['knorm']))])
            mqp=norm_back(layer.self_attn.q_norm,a['qpre'],b['qpre'],a['qnorm'],b['qnorm'],mqn,name('q_norm'))
            mkp=norm_back(layer.self_attn.k_norm,a['kpre'],b['kpre'],a['knorm'],b['knorm'],mkn,name('k_norm'))
            mqa=linear_back(layer.self_attn.q_proj,a['a'],b['a'],a['qpre'].reshape(1,n,-1),b['qpre'].reshape(1,n,-1),mqp.reshape(1,n,-1),name('q_proj'))
            mka=linear_back(layer.self_attn.k_proj,a['a'],b['a'],a['kpre'].reshape(1,n,-1),b['kpre'].reshape(1,n,-1),mkp.reshape(1,n,-1),name('k_proj'))
            mva=linear_back(layer.self_attn.v_proj,a['a'],b['a'],a['v'],b['v'],mv.transpose(1,2).reshape(1,n,-1),name('v_proj'))
            ma=mqa+mka+mva
            m_in=norm_back(layer.input_layernorm,a['x'],b['x'],a['a'],b['a'],ma,name('input_norm'))
            m=m_x_residual+m_in
            assert torch.isfinite(m).all()
            checks.append({'layer':li,'native_attention_qkv_and_output_captured':True,
                'native_LSE_probability_PV_approximation_explicit':True,'native_swiglu_product_endpoints_exact':True,
                'max_abs_input_multiplier':float(m.abs().max())})
            if progress is not None:
                progress(li,variant)
            # Release the largest matrices before loading the next layer.
            del a,b,q0,k0,v0,kr0,vr0,p0,score0,z0,q1,k1,v1,kr1,vr1,p1,score1,z1,mp,mvr,mz,mq,mkr,mk,mv
            del mg,mu,m_product,m_x_gate,m_x_up,m_mlp_input,m_norm,m_mid,m_concat,mout,mqn,mkn,mqp,mkp,mqa,mka,mva,ma,m_in,m_x_residual,silu0,silu1
        embedding_delta=f(after['layers'][0]['x'])-f(before['layers'][0]['x'])
        signed=(m.double()*embedding_delta.double()).sum(-1).flatten()
        assert torch.isfinite(signed).all()
        total=float(signed.sum())
        # Only reduced 0D FP64 tensors survive; no activation is retained here.
        # Original Python addition order is restored after one batched CPU copy.
        scalars=[]
        for _,output,incoming in pending_ledger:
            if isinstance(output,torch.Tensor):scalars.append(output)
            scalars.extend(incoming)
        values=iter(torch.stack(scalars).cpu().tolist())
        for name,output,incoming in pending_ledger:
            outgoing=next(values) if isinstance(output,torch.Tensor) else output
            incoming_credit=sum(next(values) for _ in incoming)
            ledger.append({'name':name,'outgoing_credit':outgoing,'incoming_credit':incoming_credit,'residual':outgoing-incoming_credit})
        assert next(values,None) is None
        del scalars,values,pending_ledger
        residual=math.fsum(x['residual'] for x in ledger)
        torch.cuda.synchronize()
        return {'variant':variant,'signed_full_sequence':signed.cpu().tolist(),
            'target_delta_score32_sum64':g_delta,'target_delta_score16':after['score16']-before['score16'],
            'signed_sum':total,'unassigned_total':g_delta-total,'ledger_residual_sum':residual,
            'unbooked_rounding_residual':g_delta-total-residual,
            'absolute_ledger_residual_sum':math.fsum(abs(x['residual']) for x in ledger),
            'ledger':ledger,'layer_checks':checks,'reveal_cancel_gate':rc,
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'native_forwards':0,'native_vjps':0,'manual_secant_pullback_passes':1,
            'partial_recomputation':'Per layer, two actual native FA decoder replays. Zero auxiliary attention calls. Explicit FP32 attribution P from actual QKV/LSE; original finite-softmax rule still normalizes actual QK scores. Original SiLU/product checks and complete FP64 audits retained, scalar expression compiled with existing Inductor. Native dropout0 outputs untouched.',
            'native_fa_operand_audits':fa_operand_audits,'extra_native_fa_attention_calls':0,
            'sign_scope':'Signed contrastive allocation under explicit secant/interaction rules relative to all-eligible EOS input. Not certified single-token deletion signs or global Shapley. Numerical residual is kept unassigned; negative contributions are never clipped during propagation.'}
