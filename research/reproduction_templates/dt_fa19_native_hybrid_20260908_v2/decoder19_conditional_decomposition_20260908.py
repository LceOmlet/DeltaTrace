"""Actual-coefficient conditional ledger for a captured Qwen3.5 FA decoder.

Only CPU tensor copies, reshapes, elementwise products and reductions. No model,
finite-rule, native attention, softmax, matrix multiplication or rotary call.
The output projection and sigmoid gate remain one boundary because their
pre-projection multiplier is not returned by the production diagnostics.

All terms use prediction minus actual. Coefficients are frozen from the actual
B2 pullback; deltas must be actual B1 clean minus deletion captures (or the real
B2 endpoint difference). This algebra does not prove any individual rule is a
cause. It preserves numerical boundaries and signed cancellation.
"""
import torch


D_FIELDS = ('input_norm_input', 'input_norm_output', 'post_norm_input',
            'post_norm_output', 'mlp_output', 'output')
C_FIELDS = ('input', 'output', 'q_proj_output', 'attention_output', 'query', 'key', 'value')


def _cpu(value):
    assert isinstance(value, torch.Tensor)
    return value.detach().to('cpu', copy=True)


def features(d, c, e=None):
    """Select actual native module/FA-interface captures; no e tensors needed."""
    return {'d': {name: _cpu(d[name]) for name in D_FIELDS},
            'c': {name: _cpu(c[name]) for name in C_FIELDS}}


def difference(clean, deleted):
    """Actual coordinate differences; additional captured fields are preserved."""
    assert clean.keys() == deleted.keys()
    result = {}
    for group in clean:
        assert clean[group].keys() == deleted[group].keys()
        result[group] = {}
        for name, left in clean[group].items():
            right = deleted[group][name]
            assert left.device.type == right.device.type == 'cpu' and left.shape == right.shape
            result[group][name] = left.double() - right.double()
    return result


def endpoints(captured, index):
    """Extract EOS/input rows from actual interleaved paired captures."""
    assert index in (0, 1)
    for values in captured.values():
        assert all(value.shape[0] % 2 == 0 for value in values.values())
    return {group: {name: value[index::2] for name, value in values.items()}
            for group, values in captured.items()}


def pack_coeff(upstream, new, terms):
    """Retain existing coefficients only; ignore FA tau/center debug buffers."""
    mixer = terms['mixer']
    activity = mixer['finite_FA_activity']
    assert isinstance(activity, dict), 'Require actual Vendor FA activity metadata.'
    heads, kv_heads = activity['query_heads'], activity['kv_heads']
    assert type(heads) is int and type(kv_heads) is int
    assert heads > 0 and kv_heads > 0 and heads % kv_heads == 0
    assert activity['GQA_input_expansion'] is False
    result = {'upstream': _cpu(upstream), 'input': _cpu(new)}
    result.update({name: _cpu(terms[name]) for name in
                   ('m_mlp_norm_output', 'm_mixer_output', 'm_mixer_input')})
    result.update({name: _cpu(mixer[name]) for name in ('mcontent', 'mgate')})
    result['FA_coeff'] = {name: _cpu(mixer['coeff'][name]) for name in ('dq', 'dk', 'dv')}
    result['FA_layout'] = {'query_heads': heads, 'kv_heads': kv_heads,
                           'groups': heads // kv_heads, 'padded_length': activity['padded_length'],
                           'valid_lengths': list(activity['valid_lengths']),
                           'mapping': 'query_head // groups = compact_kv_head'}
    return result


def _expect(value, shape, name):
    assert isinstance(value, torch.Tensor) and value.device.type == 'cpu', name
    assert tuple(value.shape) == tuple(shape), (name, tuple(value.shape), tuple(shape))
    assert bool(torch.isfinite(value).all()), name


def _project(coefficient, delta):
    """Both operands are [B,T,...]; preserve token locations for grouping."""
    assert coefficient.device.type == delta.device.type == 'cpu'
    assert coefficient.shape == delta.shape
    return (coefficient.double() * delta.double()).flatten(2).sum(-1)


def decompose(coeff, delta, groups=None):
    """Return a complete decoder ledger without any propagation/operator call.

    Optional groups is a disjoint partition of token positions for B=1. Groups
    describe the location of each contraction, not original source attribution.
    K/V coefficients are folded in CPU64 using the actual Vendor head metadata
    and the exact contiguous mapping used in _attention_input_rule. Production
    FP32 fold rounding is included in the combined attention-input boundary.
    """
    d, c = delta['d'], delta['c']
    u = coeff['upstream']
    assert u.ndim == 3
    B, T, width = u.shape
    meta = coeff['FA_layout']
    H, KH, G = meta['query_heads'], meta['kv_heads'], meta['groups']
    assert H == KH * G and meta['mapping'] == 'query_head // groups = compact_kv_head'
    assert meta['padded_length'] == T and meta['valid_lengths'] == [T] * B, 'Only actual equal-length unpadded captures.'
    assert coeff['mcontent'].ndim == 4
    D = coeff['mcontent'].shape[-1]
    assert D == 256, 'This ledger is for the actual BF16/D256 Vendor FA path.'
    for name in ('upstream', 'input', 'm_mlp_norm_output', 'm_mixer_output', 'm_mixer_input'):
        _expect(coeff[name], (B, T, width), 'coeff.' + name)
    _expect(coeff['mcontent'], (B, H, T, D), 'mcontent')
    _expect(coeff['mgate'], (B, T, H, D), 'mgate')
    for name in ('dq', 'dk', 'dv'):
        _expect(coeff['FA_coeff'][name], (B, H, T, D), 'FA.' + name)
    for name in D_FIELDS:
        _expect(d[name], (B, T, width), 'd.' + name)
    for name in ('input', 'output'):
        _expect(c[name], (B, T, width), 'c.' + name)
    _expect(c['q_proj_output'], (B, T, H * 2 * D), 'q_proj_output')
    _expect(c['attention_output'], (B, T, H, D), 'native_FA_attention_output')
    _expect(c['query'], (B, H, T, D), 'query')
    _expect(c['key'], (B, KH, T, D), 'compact_key')
    _expect(c['value'], (B, KH, T, D), 'compact_value')

    q = _project(coeff['FA_coeff']['dq'].transpose(1, 2), c['query'].transpose(1, 2))
    folded_k = coeff['FA_coeff']['dk'].double().reshape(B, KH, G, T, D).sum(2)
    folded_v = coeff['FA_coeff']['dv'].double().reshape(B, KH, G, T, D).sum(2)
    k = _project(folded_k.transpose(1, 2), c['key'].transpose(1, 2))
    v = _project(folded_v.transpose(1, 2), c['value'].transpose(1, 2))
    gate_delta = c['q_proj_output'].reshape(B, T, H, 2 * D)[..., D:]
    gate = _project(coeff['mgate'], gate_delta)
    attention = _project(coeff['mcontent'].transpose(1, 2), c['attention_output'])
    mm, mi, ml = (coeff[name] for name in ('m_mixer_output', 'm_mixer_input', 'm_mlp_norm_output'))
    actual = _project(u, d['output'])
    predicted = _project(coeff['input'], d['input_norm_input'])
    mlp = _project(u, d['mlp_output'])
    post_norm = _project(ml, d['post_norm_output'])
    mixer_output = _project(mm, c['output'])
    mixer_input = _project(mi, c['input'])
    terms = {
        'decoder_output_residual_rounding': _project(u, d['post_norm_input']) + mlp - actual,
        'MLP_combined': post_norm - mlp,
        'post_RMSNorm': _project(mm.double() - u.double(), d['post_norm_input']) - post_norm,
        'mixer_residual_rounding': _project(mm, d['input_norm_input']) + mixer_output - _project(mm, d['post_norm_input']),
        'attention_output_projection_and_sigmoid_gate': attention + gate - mixer_output,
        'finite_FA_core_including_seed_cast': q + k + v - attention,
        'attention_input_QKnorm_RoPE_GQA_projections': mixer_input - q - k - v - gate,
        'mixer_input_alias': _project(mi, d['input_norm_output']) - mixer_input,
        'input_RMSNorm': _project(coeff['input'].double() - mm.double(), d['input_norm_input']) - _project(mi, d['input_norm_output']),
    }
    token_residual = sum(terms.values()) - (predicted - actual)
    assert bool(torch.isfinite(token_residual).all())
    token_max = float(token_residual.abs().max())
    assert token_max < 1e-7, ('token ledger closure', token_max)
    values = {name: float(value.sum()) for name, value in terms.items()}
    output_value, input_value = float(actual.sum()), float(predicted.sum())
    error = input_value - output_value
    closure = sum(values.values()) - error
    assert abs(closure) < 1e-7, ('total ledger closure', closure)
    result = {
        'sign_convention': 'prediction_minus_actual', 'output_contraction': output_value,
        'input_contraction': input_value, 'prediction_minus_actual': error,
        'actual_minus_predicted': -error, 'terms': values, 'telescoping_error': closure,
        'max_token_telescoping_error': token_max,
        'branch_contractions': {name: float(value.sum()) for name, value in
                                {'q': q, 'k': k, 'v': v, 'attention_output': attention,
                                 'sigmoid_gate': gate, 'mixer_output': mixer_output}.items()},
        'GQA_contract': dict(meta, audit_fold_dtype='float64', production_fold_dtype='float32'),
        'scope': 'Whole FA decoder conditional ledger. Core term includes the existing mcontent-to-BF16 seed boundary. Output projection+gate and attention input remain combined; no unobserved coefficient was reconstructed.'}
    if groups is not None:
        assert B == 1 and isinstance(groups, dict) and groups
        positions = [position for indices in groups.values() for position in indices]
        assert all(type(position) is int for position in positions)
        assert len(positions) == T and set(positions) == set(range(T)), 'Groups must partition real token positions.'
        result['token_groups'] = {}
        for name, indices in groups.items():
            group_terms = {term: float(value[0, indices].sum()) for term, value in terms.items()}
            group_actual, group_prediction = float(actual[0, indices].sum()), float(predicted[0, indices].sum())
            group_error = group_prediction - group_actual
            assert abs(sum(group_terms.values()) - group_error) < 1e-7
            result['token_groups'][name] = {'count': len(indices), 'terms': group_terms,
                                           'prediction_minus_actual': group_error}
        assert all(abs(sum(g['terms'][name] for g in result['token_groups'].values()) - value) < 1e-7
                   for name, value in values.items())
        result['group_contract'] = 'Update/contraction positions, not independent source-token causal effects.'
    return result
