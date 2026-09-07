from compiled_secant_batched_layout import attention_layout
from compiled_secant_boundaries import midpoint, scaled_probability, norm_residual_two, norm_residual_three
from compiled_swiglu_secant import swiglu_secant_multipliers
"""Original active rescale finite attribution with explicit optional diagnostics omitted.
Full per-operator diagnostic reference is retained separately and never free.
"""
from compiled_native_probability_audit import compiled_pv_audit
from native_half_linear import native_half_linear
import math
import time
from compiled_logprob_seed import logprob_secant_seed
from compiled_finite_rules import softmax_secant_pullback, rmsnorm_secant_pullback

def propagate_signed_secant(model, before, after, variant='rescale', progress=None, pv_rule='symmetric'):
    import torch
    assert variant == 'rescale'
    assert pv_rule in ['symmetric', 'content_P1', 'content_P0']
    assert before['length'] == after['length'] and before['prompt_len'] == after['prompt_len']
    assert torch.equal(before['target'], after['target'])
    assert torch.equal(before['cos'], after['cos']) and torch.equal(before['sin'], after['sin'])
    assert before['mask'] is None and after['mask'] is None
    batch_size=before['last'].shape[0]
    assert batch_size==len(before['prompt_len'])==len(before['actual_lengths'])
    device = next(model.parameters()).device
    native_dtype = next(model.parameters()).dtype
    assert native_dtype == torch.float16
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
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
        reconstructed_q, reconstructed_k = apply_rotary_pos_emb(values['qnorm'].transpose(1, 2), values['knorm'].transpose(1, 2), before['cos'], before['sin'])
        assert torch.equal(reconstructed_q, qs.transpose(1, 2)) and torch.equal(reconstructed_k, ks.transpose(1, 2))
        assert torch.equal(values['v'].view(batch_size, n, -1, h), vs)
        assert torch.equal(values['fa_out'].reshape_as(values['concat']), values['concat'])
        lse = values['fa_lse'].float()
        q, k, v = (qs.transpose(1, 2).float(), ks.transpose(1, 2).float(), vs.transpose(1, 2).float())
        kr = k.repeat_interleave(groups, dim=1)
        vr = v.repeat_interleave(groups, dim=1)
        qk_unscaled = q @ kr.transpose(-1, -2)
        assert lse.shape == q.shape[:3] and lse.dtype == torch.float32
        z, prob, audit = scaled_probability(qk_unscaled, lse, layer.self_attn.scaling)
        valid, probability_relative, row_error = audit.cpu().tolist()
        assert valid == 1.0, 'Native-LSE attribution probability validity failed'
        pv = (prob @ vr).transpose(1, 2).reshape_as(values['concat'])
        actual = values['concat'].float()
        pv_relative, pv_maximum = compiled_pv_audit(pv, actual).cpu().tolist()
        fa_operand_audits.append({'layer': layer.self_attn.layer_idx, 'endpoint': 'before' if len(fa_operand_audits) % 2 == 0 else 'after', 'actual_native_qkv_and_output_exact': True, 'auxiliary_dropout': None, 'model_dropout': 0.0, 'auxiliary_output_used_by_model': False, 'extra_native_attention_calls': 0, 'probability_source': 'Explicit FP32 exp(QK*scale minus actual native FA LSE), causal masked. Attribution operand only, not a native returned matrix or model output.', 'probability_relative_l2_to_explicit_QK': probability_relative, 'probability_row_sum_max_error': row_error, 'PV_relative_l2_to_actual_FA': pv_relative, 'PV_max_error_to_actual_FA': pv_maximum})
        assert fa_operand_audits[-1]['probability_relative_l2_to_explicit_QK'] <= 0.001
        assert fa_operand_audits[-1]['probability_row_sum_max_error'] <= 0.001
        assert fa_operand_audits[-1]['PV_relative_l2_to_actual_FA'] <= 0.001
        return (q, k, v, kr, vr, prob, None, z) # Unused raw-score slot deliberately unmaterialized.
    with torch.no_grad():
        z0 = f(before['logits'])
        z1 = f(after['logits'])
        seed = logprob_secant_seed(z0, z1, after['target'].to(device))
        g_delta = after['score32_sum64'] - before['score32_sum64']
        m_final = torch.zeros_like(f(before['norm_out']))
        # Concatenated valid response logits have independent row seeds; no padded loss.
        final_seed=native_half_linear(seed, model.lm_head.weight)
        offset=0
        for sample,(plen,length) in enumerate(zip(before['prompt_len'],before['actual_lengths'])):
            count=length-plen
            m_final[sample,plen-1:length-1]=final_seed[offset:offset+count]
            offset+=count
        assert offset==final_seed.shape[0]
        del final_seed
        del seed, z0, z1
        m = norm_back(model.model.norm, f(before['last']), f(after['last']), m_final)
        del m_final
        for li in reversed(range(len(model.model.layers))):
            layer = model.model.layers[li]
            raw0 = before['layers'][li]
            raw1 = after['layers'][li]
            a = {k: f(raw0[k]) for k in ['kpre', 'mid', 'qpre', 'x']}
            b = {k: f(raw1[k]) for k in ['kpre', 'mid', 'qpre', 'x']}
            name = lambda part: f'layer{li}.{part}'
            m_mid = m
            m_product = linear_back(layer.mlp.down_proj, m)
            silu0 = torch.nn.functional.silu(raw0['gate'].to(device)).float()
            silu1 = torch.nn.functional.silu(raw1['gate'].to(device)).float()
            for raw, silu in [(raw0, silu0), (raw1, silu1)]:
                product = silu.to(native_dtype) * raw['up'].to(device)
                assert torch.equal(product, raw['product'].to(device)), 'Native SwiGLU product mismatch'
            mu, m_gate = swiglu_secant_multipliers(raw0['gate'], raw1['gate'], raw0['up'], raw1['up'], silu0, silu1, m_product)
            m_x_up = linear_back(layer.mlp.up_proj, mu)
            m_x_gate = linear_back(layer.mlp.gate_proj, m_gate)
            del m_gate
            m_mid = norm_residual_two(a['mid'], b['mid'], f(layer.post_attention_layernorm.weight), m_x_gate, m_x_up, m_mid, layer.post_attention_layernorm.variance_epsilon)
            m_x_residual = m_mid
            m_concat = linear_back(layer.self_attn.o_proj, m_mid)
            q0, k0, v0, kr0, vr0, p0, score0, z0 = native_attention_operands(raw0, layer)
            q1, k1, v1, kr1, vr1, p1, score1, z1 = native_attention_operands(raw1, layer)
            n = before['length']
            heads = q0.shape[1]
            h = layer.self_attn.head_dim
            groups = layer.self_attn.num_key_value_groups
            mout = m_concat.view(batch_size, n, heads, h).transpose(1, 2)
            if pv_rule == 'symmetric':
                mp = mout @ midpoint(vr0, vr1).transpose(-1, -2)
                mvr = midpoint(p0, p1).transpose(-1, -2) @ mout
            elif pv_rule == 'content_P1':
                mp = mout @ vr0.transpose(-1, -2)
                mvr = p1.transpose(-1, -2) @ mout
            else: # content_P0: reverse interaction allocation order.
                mp = mout @ vr1.transpose(-1, -2)
                mvr = p0.transpose(-1, -2) @ mout
            mz = softmax_secant_pullback(z0, z1, mp)
            scaling = layer.self_attn.scaling
            q_product = mz @ midpoint(kr0, kr1)
            k_product = mz.transpose(-1, -2) @ midpoint(q0, q1)
            mqn, mkn, mv = attention_layout(q_product, k_product, mvr, before['cos'], before['sin'], groups, scaling, k0.shape[1], v0.shape[1], n, h)
            mqp = norm_back(layer.self_attn.q_norm, a['qpre'], b['qpre'], mqn)
            mkp = norm_back(layer.self_attn.k_norm, a['kpre'], b['kpre'], mkn)
            mqa = linear_back(layer.self_attn.q_proj, mqp.reshape(batch_size, n, -1))
            mka = linear_back(layer.self_attn.k_proj, mkp.reshape(batch_size, n, -1))
            mva = linear_back(layer.self_attn.v_proj, mv.transpose(1, 2).reshape(batch_size, n, -1))
            m = norm_residual_three(a['x'], b['x'], f(layer.input_layernorm.weight), mqa, mka, mva, m_x_residual, layer.input_layernorm.variance_epsilon)
            assert torch.isfinite(m).all()
            checks.append({'layer': li, 'native_attention_qkv_and_output_captured': True, 'native_LSE_probability_PV_approximation_explicit': True, 'native_swiglu_product_endpoints_exact': True, 'max_abs_input_multiplier': float(m.abs().max())})
            if progress is not None:
                progress(li, variant)
            del a, b, q0, k0, v0, kr0, vr0, p0, score0, z0, q1, k1, v1, kr1, vr1, p1, score1, z1, mp, mvr, mz, q_product, k_product, mv
            del mu, m_product, m_x_gate, m_x_up, m_mid, m_concat, mout, mqn, mkn, mqp, mkp, mqa, mka, mva, m_x_residual, silu0, silu1
        embedding_delta = f(after['layers'][0]['x']) - f(before['layers'][0]['x'])
        signed = (m.double() * embedding_delta.double()).sum(-1)
        assert torch.isfinite(signed).all()
        total = signed.sum(-1).cpu()
        torch.cuda.synchronize()
        return {'variant': variant, 'pv_rule': pv_rule, 'pv_rule_scope': 'Only PV interaction allocation differs: symmetric, deltaP*V0+P1*deltaV, or deltaP*V1+P0*deltaV. Mixed endpoints are algebraic attribution terms, never claimed as actual model counterfactual forwards. NativeFA probability approximation/global residual remain explicit.', 'signed_full_sequence': signed.cpu().tolist(), 'target_delta_score32_sum64': g_delta.tolist(), 'target_delta_score16': (after['score16'] - before['score16']).tolist(), 'signed_sum': total.tolist(), 'unassigned_total': (g_delta - total).tolist(), 'ledger_residual_sum': None, 'unbooked_rounding_residual': None, 'absolute_ledger_residual_sum': None, 'ledger': None, 'layer_checks': checks, 'reveal_cancel_gate': rc, 'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'native_forwards': 0, 'native_vjps': 0, 'manual_secant_pullback_passes': 1, 'partial_recomputation': 'Same original native endpoint/replay and rescale finite attribution. Per-operator diagnostic-only FP64 credits and unused FP32 diagnostic copies omitted explicitly. Native model, FA and attribution multipliers unchanged; no surrogate gradient.', 'native_fa_operand_audits': fa_operand_audits, 'extra_native_fa_attention_calls': 0, 'sign_scope': 'Signed contrastive allocation under explicit secant/interaction rules relative to all-eligible EOS input. Not certified single-token deletion signs or global Shapley. Numerical residual is kept unassigned; negative contributions are never clipped during propagation.', 'per_operator_ledger_collected': False, 'diagnostic_scope': 'Production path: no per-operator FP64 ledger. Actual target difference, signed sum, unassigned total and native/finite-rule numerical validity checks remain. Full diagnostic reference exists and its research executions are charged separately; omitted fields are None, never zero.'}
