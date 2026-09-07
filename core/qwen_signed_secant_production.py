"""Original active rescale finite attribution with explicit optional diagnostics omitted.
Full per-operator diagnostic reference is retained separately and never free.
"""
from compiled_native_probability_audit import compiled_probability, compiled_pv_audit
from native_half_linear import native_half_linear
import math
import time
from compiled_logprob_seed import logprob_secant_seed
from compiled_finite_rules import softmax_secant_pullback, rmsnorm_secant_pullback

def propagate_signed_secant(model, before, after, variant='rescale', progress=None):
    import torch
    assert variant == 'rescale'
    assert before['length'] == after['length'] and before['prompt_len'] == after['prompt_len']
    assert torch.equal(before['target'], after['target'])
    assert torch.equal(before['cos'], after['cos']) and torch.equal(before['sin'], after['sin'])
    assert before['mask'] is None and after['mask'] is None
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
        assert torch.equal(values['v'].view(1, n, -1, h), vs)
        assert torch.equal(values['fa_out'].reshape_as(values['concat']), values['concat'])
        lse = values['fa_lse'].float()
        q, k, v = (qs.transpose(1, 2).float(), ks.transpose(1, 2).float(), vs.transpose(1, 2).float())
        kr = k.repeat_interleave(groups, dim=1)
        vr = v.repeat_interleave(groups, dim=1)
        raw = q @ kr.transpose(-1, -2) * layer.self_attn.scaling
        assert lse.shape == q.shape[:3] and lse.dtype == torch.float32
        z, prob, audit = compiled_probability(raw, lse)
        valid, probability_relative, row_error = audit.cpu().tolist()
        assert valid == 1.0, 'Native-LSE attribution probability validity failed'
        pv = (prob @ vr).transpose(1, 2).reshape_as(values['concat'])
        actual = values['concat'].float()
        pv_relative, pv_maximum = compiled_pv_audit(pv, actual).cpu().tolist()
        fa_operand_audits.append({'layer': layer.self_attn.layer_idx, 'endpoint': 'before' if len(fa_operand_audits) % 2 == 0 else 'after', 'actual_native_qkv_and_output_exact': True, 'auxiliary_dropout': None, 'model_dropout': 0.0, 'auxiliary_output_used_by_model': False, 'extra_native_attention_calls': 0, 'probability_source': 'Explicit FP32 exp(QK*scale minus actual native FA LSE), causal masked. Attribution operand only, not a native returned matrix or model output.', 'probability_relative_l2_to_explicit_QK': probability_relative, 'probability_row_sum_max_error': row_error, 'PV_relative_l2_to_actual_FA': pv_relative, 'PV_max_error_to_actual_FA': pv_maximum})
        assert fa_operand_audits[-1]['probability_relative_l2_to_explicit_QK'] <= 0.001
        assert fa_operand_audits[-1]['probability_row_sum_max_error'] <= 0.001
        assert fa_operand_audits[-1]['PV_relative_l2_to_actual_FA'] <= 0.001
        return (q, k, v, kr, vr, prob, raw, z)
    with torch.no_grad():
        z0 = f(before['logits'])
        z1 = f(after['logits'])
        seed = logprob_secant_seed(z0, z1, after['target'].to(device))
        g_delta = after['score32_sum64'] - before['score32_sum64']
        start = before['prompt_len'] - 1
        m_final = torch.zeros_like(f(before['norm_out']))
        m_final[0, start:-1] = native_half_linear(seed, model.lm_head.weight)
        del seed, z0, z1
        m = norm_back(model.model.norm, f(before['last']), f(after['last']), m_final)
        del m_final
        for li in reversed(range(len(model.model.layers))):
            layer = model.model.layers[li]
            raw0 = before['layers'][li]
            raw1 = after['layers'][li]
            a = {k: f(raw0[k]) for k in ['gate', 'kpre', 'mid', 'qpre', 'up', 'x']}
            b = {k: f(raw1[k]) for k in ['gate', 'kpre', 'mid', 'qpre', 'up', 'x']}
            name = lambda part: f'layer{li}.{part}'
            m_mid = m
            m_product = linear_back(layer.mlp.down_proj, m)
            silu0 = torch.nn.functional.silu(raw0['gate'].to(device)).float()
            silu1 = torch.nn.functional.silu(raw1['gate'].to(device)).float()
            for raw, silu in [(raw0, silu0), (raw1, silu1)]:
                product = silu.to(native_dtype) * raw['up'].to(device)
                assert torch.equal(product, raw['product'].to(device)), 'Native SwiGLU product mismatch'
            mg = m_product * (a['up'] + b['up']) * 0.5
            mu = m_product * (silu0 + silu1) * 0.5
            m_x_up = linear_back(layer.mlp.up_proj, mu)
            difference = b['gate'] - a['gate']
            sigmoid = torch.sigmoid(a['gate'])
            derivative = sigmoid * (1 + a['gate'] * (1 - sigmoid))
            multiplier = torch.where(difference != 0, (silu1 - silu0) / torch.where(difference != 0, difference, torch.ones_like(difference)), derivative)
            m_gate = mg * multiplier
            m_x_gate = linear_back(layer.mlp.gate_proj, m_gate)
            del difference, sigmoid, derivative, multiplier, m_gate
            m_mlp_input = m_x_gate + m_x_up
            m_norm = norm_back(layer.post_attention_layernorm, a['mid'], b['mid'], m_mlp_input)
            m_mid = m_mid + m_norm
            m_x_residual = m_mid
            m_concat = linear_back(layer.self_attn.o_proj, m_mid)
            q0, k0, v0, kr0, vr0, p0, score0, z0 = native_attention_operands(raw0, layer)
            q1, k1, v1, kr1, vr1, p1, score1, z1 = native_attention_operands(raw1, layer)
            n = before['length']
            heads = q0.shape[1]
            h = layer.self_attn.head_dim
            groups = layer.self_attn.num_key_value_groups
            mout = m_concat.view(1, n, heads, h).transpose(1, 2)
            mp = mout @ ((vr0 + vr1) * 0.5).transpose(-1, -2)
            mvr = ((p0 + p1) * 0.5).transpose(-1, -2) @ mout
            mz = softmax_secant_pullback(z0, z1, mp)
            scaling = layer.self_attn.scaling
            mq = mz @ ((kr0 + kr1) * 0.5) * scaling
            mkr = mz.transpose(-1, -2) @ ((q0 + q1) * 0.5) * scaling
            mk = mkr.reshape(1, k0.shape[1], groups, n, h).sum(2)
            mv = mvr.reshape(1, v0.shape[1], groups, n, h).sum(2)
            cos = f(before['cos']).unsqueeze(1)
            sin = f(before['sin']).unsqueeze(1)
            mqn = rotation_transpose(mq, cos, sin).transpose(1, 2)
            mkn = rotation_transpose(mk, cos, sin).transpose(1, 2)
            mqp = norm_back(layer.self_attn.q_norm, a['qpre'], b['qpre'], mqn)
            mkp = norm_back(layer.self_attn.k_norm, a['kpre'], b['kpre'], mkn)
            mqa = linear_back(layer.self_attn.q_proj, mqp.reshape(1, n, -1))
            mka = linear_back(layer.self_attn.k_proj, mkp.reshape(1, n, -1))
            mva = linear_back(layer.self_attn.v_proj, mv.transpose(1, 2).reshape(1, n, -1))
            ma = mqa + mka + mva
            m_in = norm_back(layer.input_layernorm, a['x'], b['x'], ma)
            m = m_x_residual + m_in
            assert torch.isfinite(m).all()
            checks.append({'layer': li, 'native_attention_qkv_and_output_captured': True, 'native_LSE_probability_PV_approximation_explicit': True, 'native_swiglu_product_endpoints_exact': True, 'max_abs_input_multiplier': float(m.abs().max())})
            if progress is not None:
                progress(li, variant)
            del a, b, q0, k0, v0, kr0, vr0, p0, score0, z0, q1, k1, v1, kr1, vr1, p1, score1, z1, mp, mvr, mz, mq, mkr, mk, mv
            del mg, mu, m_product, m_x_gate, m_x_up, m_mlp_input, m_norm, m_mid, m_concat, mout, mqn, mkn, mqp, mkp, mqa, mka, mva, ma, m_in, m_x_residual, silu0, silu1
        embedding_delta = f(after['layers'][0]['x']) - f(before['layers'][0]['x'])
        signed = (m.double() * embedding_delta.double()).sum(-1).flatten()
        assert torch.isfinite(signed).all()
        total = float(signed.sum())
        torch.cuda.synchronize()
        return {'variant': variant, 'signed_full_sequence': signed.cpu().tolist(), 'target_delta_score32_sum64': g_delta, 'target_delta_score16': after['score16'] - before['score16'], 'signed_sum': total, 'unassigned_total': g_delta - total, 'ledger_residual_sum': None, 'unbooked_rounding_residual': None, 'absolute_ledger_residual_sum': None, 'ledger': None, 'layer_checks': checks, 'reveal_cancel_gate': rc, 'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'native_forwards': 0, 'native_vjps': 0, 'manual_secant_pullback_passes': 1, 'partial_recomputation': 'Same original native endpoint/replay and rescale finite attribution. Per-operator diagnostic-only FP64 credits and unused FP32 diagnostic copies omitted explicitly. Native model, FA and attribution multipliers unchanged; no surrogate gradient.', 'native_fa_operand_audits': fa_operand_audits, 'extra_native_fa_attention_calls': 0, 'sign_scope': 'Signed contrastive allocation under explicit secant/interaction rules relative to all-eligible EOS input. Not certified single-token deletion signs or global Shapley. Numerical residual is kept unassigned; negative contributions are never clipped during propagation.', 'per_operator_ledger_collected': False, 'diagnostic_scope': 'Production path: no per-operator FP64 ledger. Actual target difference, signed sum, unassigned total and native/finite-rule numerical validity checks remain. Full diagnostic reference exists and its research executions are charged separately; omitted fields are None, never zero.'}
