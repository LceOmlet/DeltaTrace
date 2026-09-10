from compiled_secant_boundaries import norm_residual_three
from qwen3_cast_boundaries import attention_layout,norm_residual_two,norm as cast_final_norm
from compiled_swiglu_secant import swiglu_secant_multipliers
"""Original active rescale finite attribution with explicit optional diagnostics omitted.
Full per-operator diagnostic reference is retained separately and never free.
"""
from native_half_linear import native_half_linear
import math
import time
from compiled_logprob_seed import compiled_seed
from compiled_finite_rules import rmsnorm_secant_pullback

def propagate_signed_secant(model, before, after, variant='rescale', progress=None, pv_rule='content_P1', finite_attention=None, finite_activity=None,validation=None):
    assert validation is not None
    import torch
    assert variant == 'rescale'
    assert pv_rule == 'content_P1' and callable(finite_attention)
    if finite_activity is None: finite_activity=[]
    assert before['length'] == after['length'] and before['prompt_len'] == after['prompt_len']
    assert before['mask'] is None and after['mask'] is None
    device = next(model.parameters()).device
    native_dtype = next(model.parameters()).dtype
    assert native_dtype == torch.float16
    started = time.perf_counter()
    checks = []
    rc = []
    fa_operand_audits = []

    def f(x):
        return x.to(device=device, dtype=torch.float32)

    def norm_back(module, x0, x1, m):
        answer = rmsnorm_secant_pullback(x0, x1, f(module.weight), m, module.variance_epsilon)
        return answer

    def linear_back(module, m):
        answer = native_half_linear(m, module.weight)
        return answer

    def rotate_half(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def rotation_transpose(m, cos, sin):
        return m * cos - rotate_half(m * sin)

    def native_attention_operands(values, layer):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        qs, ks, vs = (values['fa_q'], values['fa_k'], values['fa_v'])
        n = before['length']
        h = layer.self_attn.head_dim
        groups = layer.self_attn.num_key_value_groups
        validation.rope(values['qnorm'],values['knorm'],qs,ks,before['cos'],before['sin'],
            f'layer{layer.self_attn.layer_idx}.rope_q',f'layer{layer.self_attn.layer_idx}.rope_k')
        validation.equal(values['v'].view(1,n,-1,h),vs,f'layer{layer.self_attn.layer_idx}.value')
        validation.equal(values['fa_out'].reshape_as(values['concat']),values['concat'],f'layer{layer.self_attn.layer_idx}.concat')
        lse = values['fa_lse']
        q, k, v = qs.transpose(1,2), ks.transpose(1,2), vs.transpose(1,2)
        assert q.dtype==k.dtype==v.dtype==torch.float16 and h==128
        assert lse.shape==q.shape[:3] and lse.dtype==torch.float32
        validation.finite(lse,f'layer{layer.self_attn.layer_idx}.lse')
        kr=k;vr=v  # Compact GQA inputs; kernel preserves the original query-head outputs.
        fa_operand_audits.append({'layer':layer.self_attn.layer_idx,
            'endpoint':'before' if len(fa_operand_audits)%2==0 else 'after',
            'actual_native_qkv_and_output_exact':True,'model_dropout':0.0,
            'auxiliary_output_used_by_model':False,'extra_native_attention_calls':0,
            'probability_source':'No global probability matrix. Finite FA extension reconstructs native-LSE probabilities only within tiles.',
            'dense_probability_audit_performed':False,
            'audit_scope':'Exact original RoPE/QKV/output provenance plus actual native public LSE; dense numerical audit is a separate charged development control.'})
        return q,k,v,kr,vr,lse
    with torch.no_grad():
        z0 = f(before['logits'])
        z1 = f(after['logits'])
        seed, seed_check = compiled_seed(z0, z1, after['target'].to(device))
        validation.predicate(seed_check,'finite_logprob_seed')
        g_delta = after['score32_sum64'] - before['score32_sum64']
        start = before['prompt_len'] - 1
        m_final = torch.zeros_like(before['norm_out'],device=device,dtype=torch.float32)
        m_final[0, start:-1] = native_half_linear(seed, model.lm_head.weight)
        del seed, z0, z1
        m = cast_final_norm(before['last'],after['last'],model.model.norm.weight,m_final,model.model.norm.variance_epsilon)
        del m_final
        for li in reversed(range(len(model.model.layers))):
            layer = model.model.layers[li]
            raw0 = before['layers'][li]
            raw1 = after['layers'][li]
            if progress is not None:progress(li,variant,'native_ready')
            a = {'mid':raw0['mid']}
            b = {'mid':raw1['mid']}
            name = lambda part: f'layer{li}.{part}'
            m_mid = m
            m_product = linear_back(layer.mlp.down_proj, m)
            silu0 = torch.nn.functional.silu(raw0['gate'].to(device)).float()
            silu1 = torch.nn.functional.silu(raw1['gate'].to(device)).float()
            for raw, silu in [(raw0, silu0), (raw1, silu1)]:
                product = silu.to(native_dtype) * raw['up'].to(device)
                validation.equal(product,raw['product'].to(device),f'layer{li}.swiglu')
            mu, m_gate = swiglu_secant_multipliers(raw0['gate'], raw1['gate'], raw0['up'], raw1['up'], silu0, silu1, m_product)
            del m_product,silu0,silu1,raw,silu,product
            m_x_up = linear_back(layer.mlp.up_proj, mu)
            del mu
            m_x_gate = linear_back(layer.mlp.gate_proj, m_gate)
            del m_gate
            m_mid = norm_residual_two(a['mid'], b['mid'], layer.post_attention_layernorm.weight, m_x_gate, m_x_up, m_mid, layer.post_attention_layernorm.variance_epsilon)
            del m_x_gate,m_x_up
            a.clear();b.clear()
            if progress is not None:progress(li,variant,'after_mlp')
            m_x_residual = m_mid
            m_concat = linear_back(layer.self_attn.o_proj, m_mid)
            q0,k0,v0,kr0,vr0,lse0=native_attention_operands(raw0,layer)
            q1,k1,v1,kr1,vr1,lse1=native_attention_operands(raw1,layer)
            n=before['length'];heads=q0.shape[1];h=layer.self_attn.head_dim
            groups=layer.self_attn.num_key_value_groups
            mout=m_concat.view(1,n,heads,h).transpose(1,2)
            activity={'layer':li};finite_activity.append(activity)
            with torch.profiler.record_function('ATTR_VENDOR_FA_FINITE_P1'):
                outputs=finite_attention({'q0':q0,'k0':kr0,'q1':q1,'k1':kr1,
                    'v0':vr0,'u':mout,'lse0':lse0,'lse1':lse1},layer.self_attn.scaling,activity)
            for key,value in outputs.items():validation.finite(value,f'layer{li}.finite_FA_{key}')
            validation.positive(outputs['tau'],f'layer{li}.tau')
            # Kernel Q/K outputs already include the attention scale. Existing
            # compiled layout applies unit scale, and sums GQA in FP32.
            q_product=outputs['dq'];k_product=outputs['dk'];mvr=outputs['dv']
            mqn,mkn,mv=attention_layout(q_product,k_product,mvr,before['cos'],before['sin'],
                groups,1.0,k0.shape[1],v0.shape[1],n,h)
            mqp = norm_back(layer.self_attn.q_norm, f(raw0['qpre']), f(raw1['qpre']), mqn)
            mkp = norm_back(layer.self_attn.k_norm, f(raw0['kpre']), f(raw1['kpre']), mkn)
            mqa = linear_back(layer.self_attn.q_proj, mqp.reshape(1, n, -1))
            mka = linear_back(layer.self_attn.k_proj, mkp.reshape(1, n, -1))
            mva = linear_back(layer.self_attn.v_proj, mv.transpose(1, 2).reshape(1, n, -1))
            m = norm_residual_three(f(raw0['x']), f(raw1['x']), f(layer.input_layernorm.weight), mqa, mka, mva, m_x_residual, layer.input_layernorm.variance_epsilon)
            validation.finite(m,f'layer{li}.multiplier')
            checks.append({'layer': li, 'native_attention_qkv_and_output_captured': True, 'native_LSE_probability_reconstructed_only_in_finite_tiles': True, 'native_swiglu_product_endpoints_exact': True, 'max_abs_input_multiplier': validation.max_abs(m)})
            if progress is not None:
                progress(li, variant)
            del a,b,q0,k0,v0,kr0,vr0,lse0,q1,k1,v1,kr1,vr1,lse1,mvr,q_product,k_product,mv,outputs
            del raw0,raw1
            del m_mid, m_concat, mout, mqn, mkn, mqp, mkp, mqa, mka, mva, m_x_residual
        embedding_delta = f(after['layers'][0]['x']) - f(before['layers'][0]['x'])
        signed = (m.double() * embedding_delta.double()).sum(-1).flatten()
        total = signed.sum()
        return {'variant': variant, 'pv_rule': pv_rule, 'finite_attention_activity':finite_activity, 'pv_rule_scope': 'Content P1 only: deltaP*V0+P1*deltaV; finite logmean softmax and symmetric QK allocation. Half precision and actual public native LSE are explicit numerical choices. Mixed endpoints are algebraic terms, never claimed model counterfactual forwards.', 'signed_full_sequence': signed, 'target_delta_score32_sum64': g_delta, 'target_delta_score16': after['score16'] - before['score16'], 'signed_sum': total, 'unassigned_total': g_delta - total, 'ledger_residual_sum': None, 'unbooked_rounding_residual': None, 'absolute_ledger_residual_sum': None, 'ledger': None, 'layer_checks': checks, 'reveal_cancel_gate': rc, 'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'native_forwards': 0, 'native_vjps': 0, 'manual_secant_pullback_passes': 1, 'partial_recomputation': 'Original native endpoint/replay preserved. Content-P1 finite attention uses traceable FA-framework extension with FP16 operands/outputs and FP32 accumulators. Remaining finite propagation unchanged. No dense probability/score/finite audit matrices in production. No model or native-autograd substitution.', 'native_fa_operand_audits': fa_operand_audits, 'extra_native_fa_attention_calls': 0, 'sign_scope': 'Signed contrastive allocation under explicit secant/interaction rules relative to all-eligible EOS input. Not certified single-token deletion signs or global Shapley. Numerical residual is kept unassigned; negative contributions are never clipped during propagation.', 'per_operator_ledger_collected': False, 'diagnostic_scope': 'Production path: no per-operator FP64 ledger. Actual target difference, signed sum, unassigned total and native/finite-rule numerical validity checks remain. Full diagnostic reference exists and its research executions are charged separately; omitted fields are None, never zero.'}
