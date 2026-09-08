"""One saved-layer candidate call, then CPU-only fixed conditional ledgers.

Reuse the actual prior native B2 LSE; zero new FA/model/DT/scorer/backward calls.
The observations cannot establish a new MAS/RISE result or production speed.
"""
import gc
import hashlib
import json
import signal
import time
import traceback
import zipfile
from pathlib import Path

A = Path(__file__).resolve().parent
p = json.loads((A / 'protocol.json').read_bytes())
started = time.perf_counter()
r = {'status': 'starting', 'protocol': p, 'native_FA_calls': 0, 'model_calls': 0,
     'DT_calls': 0, 'scorer_calls': 0, 'FT_calls': 0, 'backward_calls': 0, 'points': {}}
torch = op = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save():
    target = A / 'results.partial'
    target.write_text(json.dumps(r, indent=2, allow_nan=False))
    target.replace(A / 'results.json')


def timeout(*args):
    raise TimeoutError('Frozen one-candidate saved-layer budget exceeded.')


def cpu(value):
    return value.detach().to('cpu', copy=True)


def stats(value):
    value = value.double()
    assert bool(torch.isfinite(value).all())
    return {'net': float(value.sum()), 'positive_sum': float(value.clamp_min(0).sum()),
            'negative_sum': float(value.clamp_max(0).sum()), 'absolute_sum': float(value.abs().sum())}


def coefficient_effect(coeff, delta, name):
    multiplier = coeff[name].double()
    if name != 'dq':
        multiplier = multiplier.reshape(1, KH, G, T, D).sum(2)
    assert multiplier.shape == delta.shape
    return (multiplier * delta).sum((1, 3))


def qkv_effect(coeff, change):
    q = coefficient_effect(coeff, change['query'], 'dq')
    k = coefficient_effect(coeff, change['key'], 'dk')
    v = coefficient_effect(coeff, change['value'], 'dv')
    return q + k, v


try:
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(p['run_budget']['wall_seconds'])
    for name, want in p['files_sha256'].items():
        assert digest(A / name) == want, name
    for item in p['protected_sources']:
        assert digest(item['path']) == item['sha256'], item['path']
    build = json.loads((A / 'build_results.json').read_bytes())
    assert build['status'] == 'finite_extension_compiled_not_executed' and build['protocol'] == p
    library = A / p['diagnostic_library_name']
    assert digest(library) == build['library']['sha256']
    r['build_results_sha256'] = digest(A / 'build_results.json')
    source = json.loads(Path(p['source_results_path']).read_bytes())
    hybrid = json.loads(Path(p['hybrid_results_path']).read_bytes())
    strict = json.loads(Path(p['strict_results_path']).read_bytes())
    assert strict['status'] == 'three_FA19_conditional_score_contractions_complete'
    assert strict['diagnostic_finite_entered'] == strict['diagnostic_finite_returned'] == 3
    import numpy as np
    import torch
    from vendor_fa_finite_bf16_d256 import RightPaddedLengths
    from vendor_fa_clean_secant_20260908 import ROW_FIELDS, VendorFACleanSecant
    torch.set_num_threads(4)
    assert list(ROW_FIELDS) == p['row_fields']
    original_path = Path(p['private_artifact_path'])
    lse_path = Path(p['saved_diagnostic_artifact_path'])
    for path, key in ((original_path, 'private_artifact'), (lse_path, 'saved_diagnostic_artifact')):
        assert path.stat().st_size == p[key + '_bytes'] and digest(path) == p[key + '_sha256']
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    prior = torch.load(lse_path, map_location='cpu', weights_only=True)
    lse = prior['public_B2_LSE']
    coeff = original['layers']['19']['coeff']
    saved = coeff['FA_coeff']
    m = coeff['mcontent']
    B, H, T, D = m.shape
    KH, G = coeff['FA_layout']['kv_heads'], coeff['FA_layout']['groups']
    assert (B, H, KH, T, D) == (1, 16, 4, 351, 256) and H == KH * G
    assert lse.shape == (2, H, T) and lse.dtype == torch.float32
    assert all(torch.equal(prior['points'][step][name], saved[name])
               for step in p['fixed_steps'] for name in ('dq', 'dk', 'dv'))
    names = ('query', 'key', 'value', 'attention_output')
    paired = {name: original['layers']['19']['paired']['c'][name] for name in names}
    points = {step: {name: original['scoring'][step]['19']['c'][name] for name in names}
              for step in ('0', *p['fixed_steps'])}
    r['input'] = source['input']
    r['saved_B1_vs_B2_endpoint_drift'] = hybrid['saved_B1_vs_B2_endpoint_drift']
    del original, prior, coeff
    gc.collect()
    with np.load(p['hybrid_vectors_path'], allow_pickle=False) as archive:
        reference = {step: {'R0': torch.from_numpy(archive[step + '_R0'].copy()),
                            'qk_prediction': torch.from_numpy(archive[step + '_qk_prediction'].copy())}
                     for step in p['fixed_steps']}
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    endpoints = {name: value.to('cuda').contiguous() for name, value in paired.items() if name != 'attention_output'}
    lse_gpu = lse.to('cuda').contiguous()
    operands = {'q0': endpoints['query'][0:1], 'q1': endpoints['query'][1:2],
                'k0': endpoints['key'][0:1], 'k1': endpoints['key'][1:2],
                'v0': endpoints['value'][0:1], 'u': m.to('cuda').contiguous(),
                'lse0': lse_gpu[0:1], 'lse1': lse_gpu[1:2]}
    r['operand_metadata'] = {name: {'shape': list(value.shape), 'stride': list(value.stride()),
        'dtype': str(value.dtype), 'device': str(value.device)} for name, value in operands.items()}
    op = VendorFACleanSecant(library, build['library']['sha256'])
    lengths = RightPaddedLengths([T], T, operands['q0'].device)
    r['status'] = 'one_saved_layer_candidate_call'
    save()
    torch.cuda.synchronize()
    tick = time.perf_counter()
    result = op(operands, p['scale'], lengths)
    torch.cuda.synchronize()
    r['candidate_wrapper_seconds_including_prepare_checks_and_sync'] = time.perf_counter() - tick
    result = {name: cpu(value) for name, value in result.items()}
    torch.save(result, A / 'candidate_coefficients_private.pt')
    r['candidate_private'] = {'file': 'candidate_coefficients_private.pt',
        'sha256': digest(A / 'candidate_coefficients_private.pt'),
        'bytes': (A / 'candidate_coefficients_private.pt').stat().st_size}
    r['finite_element_counts'] = {name: {'total': value.numel(), 'finite': int(torch.isfinite(value).sum())}
                                   for name, value in result.items()}
    save()
    assert all(bool(torch.isfinite(value).all()) for value in result.values()), 'Preserved nonfinite candidate; no epsilon or fallback.'
    assert result['rows'].shape == (len(ROW_FIELDS), B, H, T)
    rows = {name: result['rows'][i].double() for i, name in enumerate(ROW_FIELDS)}
    assert bool((rows['variance_M2'] >= 0).all())
    assert bool((rows['raw_p0_sum'] > 0).all()) and bool((rows['raw_p1_sum'] > 0).all())
    r['dv_bitwise_equal_to_original'] = bool(torch.equal(result['dv'], saved['dv']))
    assert r['dv_bitwise_equal_to_original'], 'Route change must not alter the original raw-P1 content branch.'
    assert not bool(result['dq'][:, :, 0].any()), 'A one-key softmax has zero route response.'
    degenerate = rows['variance_M2'] == 0
    assert bool((rows['alpha'][degenerate] == 0).all())
    positive_variance = rows['variance_M2'][~degenerate]
    correction_norm = rows['alpha'].abs() * rows['variance_M2'].sqrt()
    row_contraction = rows['endpoint_positive'] + rows['endpoint_negative']
    normalized_target = rows['normalized_route_target']
    r['row_audit'] = {
        'state_bytes': result['rows'].numel() * result['rows'].element_size(),
        'degenerate_count': int(degenerate.sum()),
        'degenerate_target_residual': stats(normalized_target[degenerate]),
        'degenerate_target_max_absolute': float(normalized_target[degenerate].abs().max()) if bool(degenerate.any()) else 0,
        'minimum_positive_variance': float(positive_variance.min()) if positive_variance.numel() else None,
        'max_absolute_alpha': float(rows['alpha'].abs().max()),
        'max_correction_L2': float(correction_norm.max()),
        'sum_correction_L2': float(correction_norm.sum()),
        'raw_probability_sum_deviation_max': {name: float((rows[name] - 1).abs().max()) for name in ('raw_p0_sum', 'raw_p1_sum')},
        'normalization_target_change': stats(normalized_target - rows['raw_route_target']),
        'normalized_target': stats(normalized_target), 'raw_target': stats(rows['raw_route_target']),
        'base_gradient_contraction': stats(rows['base_gradient_contraction']),
        'endpoint_positive_sum': float(rows['endpoint_positive'].sum()),
        'endpoint_negative_sum': float(rows['endpoint_negative'].sum()),
        'row_FP32_constraint_residual': stats(row_contraction - normalized_target),
        'row_FP32_constraint_max_absolute': float((row_contraction - normalized_target).abs().max())}
    arrays = {name: value.numpy() for name, value in rows.items()}
    P, keep = source['input']['prompt_length'], set(source['input']['keep'])
    for step in (*p['fixed_steps'], 'B2'):
        if step == 'B2':
            change = {name: value[1:2].double() - value[0:1].double() for name, value in paired.items()}
            groups = {'eligible_prompt': sorted(keep), 'other_prompt': sorted(set(range(P)) - keep),
                      'response': list(range(P, T))}
        else:
            change = {name: points['0'][name].double() - points[step][name].double() for name in names}
            deleted = set(source['points'][step]['input_receipt']['deleted_positions'])
            assert deleted <= keep
            if step == '20':
                assert deleted == keep
            groups = {'deleted': sorted(deleted), 'kept': sorted(keep - deleted),
                      'other_prompt': sorted(set(range(P)) - keep), 'response': list(range(P, T))}
        old_qk, old_v = qkv_effect(saved, change)
        new_qk, new_v = qkv_effect(result, change)
        assert torch.equal(old_v, new_v)
        actual = (m.double().transpose(1, 2) * change['attention_output']).sum((2, 3))
        actual_bf16_seed = (m.to(torch.bfloat16).double().transpose(1, 2) * change['attention_output']).sum((2, 3))
        fields = {'old_QK': old_qk, 'candidate_QK': new_qk, 'unchanged_V': old_v,
            'actual_at_stored_seed': actual, 'actual_at_BF16_seed': actual_bf16_seed,
            'old_core_error': old_qk + old_v - actual,
            'candidate_core_error': new_qk + new_v - actual,
            'candidate_minus_old_error': new_qk - old_qk}
        if step != 'B2':
            assert float((old_qk - reference[step]['qk_prediction']).abs().max()) < 1e-7
            original_error = source['points'][step]['layer_decompositions']['19']['terms']['finite_FA_core_including_seed_cast']
            assert abs(float(fields['old_core_error'].sum()) - original_error) < 1e-7
            fields['old_route_error_at_V0'] = old_qk - reference[step]['R0']
            fields['candidate_route_error_at_V0'] = new_qk - reference[step]['R0']
        else:
            fields['normalized_route_target'] = normalized_target.sum(1)
            fields['raw_route_target'] = rows['raw_route_target'].sum(1)
            fields['candidate_QK_minus_normalized_route_target'] = new_qk - normalized_target.sum(1)
        row = {'fields': {name: stats(value) for name, value in fields.items()},
            'groups': {group: {'count': len(indices), 'fields': {name: stats(value[0, indices])
                       for name, value in fields.items()}} for group, indices in groups.items()}}
        assert sum(map(len, groups.values())) == T
        for name, value in fields.items():
            assert abs(sum(group['fields'][name]['net'] for group in row['groups'].values()) - float(value.sum())) < 1e-7
            arrays[step + '_' + name] = value.numpy()
        r['points'][step] = row
        save()
    assert op.calls_entered == op.calls_returned == 1
    np.savez_compressed(A / 'candidate_signed_rows.npz', **arrays)
    r['signed_rows'] = {'file': 'candidate_signed_rows.npz', 'sha256': digest(A / 'candidate_signed_rows.npz')}
    for item in p['protected_sources']:
        assert digest(item['path']) == item['sha256']
    assert digest(original_path) == p['private_artifact_sha256'] and digest(lse_path) == p['saved_diagnostic_artifact_sha256']
    r['status'] = 'one_clean_secant_saved_layer_candidate_observed'
except Exception:
    r['status'] = 'failed'
    r['error'] = traceback.format_exc()
finally:
    signal.alarm(0)
    r['seconds'] = time.perf_counter() - started
    r['candidate_finite_entered'] = op.calls_entered if op is not None else 0
    r['candidate_finite_returned'] = op.calls_returned if op is not None else 0
    if torch is not None and torch.cuda.is_initialized():
        r['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        r['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    r['scope'] = 'One saved FA19 candidate coefficient call and CPU fixed early/mid/allEOS/B2 contractions; not whole-layer pullback, new input attribution, new MAS/RISE or production efficiency.'
    r['limits'] = 'Route probabilities are explicitly row normalized; raw-P1 dv unchanged. Numeric normalization changes and degenerate incompatibility remain unassigned/reportable. Conservation by large opposite signed credits is not accepted as method evidence. No arbitrary-direction or metric improvement is guaranteed.'
    r['group_contract'] = 'Q/K/V contractions use operand coordinates; actual output uses query coordinates. Signs are coordinate allocations, not independent input-source causal contributions.'
    save()
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in [*p['files_sha256'], 'protocol.json', 'build_results.json', 'compile.log', 'results.json', 'candidate_signed_rows.npz']:
            if (A / name).is_file():
                bundle.write(A / name, name)
    print(json.dumps({'status': r['status'], 'seconds': r['seconds'],
        'candidate_finite_entered': r['candidate_finite_entered'],
        'candidate_finite_returned': r['candidate_finite_returned'], 'error': r.get('error')}), flush=True)
