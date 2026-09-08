"""One saved FA19 layer: three conditional score contractions, no model run.

Uses one unchanged public FA call for paired LSE, then three executions of the
isolated, source-traceable finite diagnostic. Actual hybrid R0 is frozen from
the completed eleven-call operator study. No probabilities are materialized.
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
r = {'status': 'starting', 'protocol': p, 'native_FA_entered': 0,
     'native_FA_returned': 0, 'model_calls': 0, 'DT_calls': 0, 'scorer_calls': 0,
     'FT_calls': 0, 'backward_calls': 0, 'points': {}}
torch = op = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save():
    temporary = A / 'results.partial'
    temporary.write_text(json.dumps(r, indent=2, allow_nan=False))
    temporary.replace(A / 'results.json')


def timeout(*_args):
    raise TimeoutError('Frozen single-layer diagnostic execution budget exceeded.')


def cpu(value):
    return value.detach().to('cpu', copy=True)


def layout(value):
    return {'shape': list(value.shape), 'stride': list(value.stride()),
            'dtype': str(value.dtype), 'device': str(value.device),
            'is_cuda': value.is_cuda, 'contiguous': value.is_contiguous()}


def stats(value):
    value = value.double()
    assert bool(torch.isfinite(value).all())
    return {'net': float(value.sum()), 'positive_sum': float(value.clamp_min(0).sum()),
            'negative_sum': float(value.clamp_max(0).sum()), 'absolute_sum': float(value.abs().sum())}


def drift(new, old):
    assert new.shape == old.shape and new.dtype == old.dtype
    change = new.double() - old.double()
    norm = float(old.double().norm())
    return {'bitwise_equal': bool(torch.equal(new, old)),
            'relative_L2': float(change.norm()) / norm if norm else None,
            'max_absolute_difference': float(change.abs().max())}


def prediction(coeff, delta_q, delta_k):
    dq, dk = coeff['dq'].double(), coeff['dk'].double()
    dk = dk.reshape(1, KH, G, T, D).sum(2)
    assert dq.shape == delta_q.shape and dk.shape == delta_k.shape
    return (dq * delta_q).sum((1, 3)) + (dk * delta_k).sum((1, 3))


try:
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(p['run_budget']['wall_seconds'])
    assert p['fixed_steps'] == ['1', '10', '20']
    for name, want in p['files_sha256'].items():
        assert digest(A / name) == want, name
    for item in p['protected_sources']:
        assert digest(item['path']) == item['sha256'], item['path']
    build = json.loads((A / 'build_results.json').read_bytes())
    assert build['status'] == 'finite_extension_compiled_not_executed'
    assert build['protocol'] == p
    library = A / p['diagnostic_library_name']
    assert digest(library) == build['library']['sha256']
    r['build_results_sha256'] = digest(A / 'build_results.json')
    r['diagnostic_library_sha256'] = digest(library)
    source = json.loads(Path(p['source_results_path']).read_bytes())
    hybrid = json.loads(Path(p['hybrid_results_path']).read_bytes())
    assert source['status'] == 'decoder19_6_actual_conditional_observation_complete'
    assert hybrid['status'] == 'eleven_native_FA_hybrid_contrasts_complete'

    import numpy as np
    import torch
    import flash_attn
    import flash_attn.flash_attn_interface as fa
    from vendor_fa_finite_bf16_d256 import RightPaddedLengths
    from vendor_fa_conditional_diag_20260908 import VendorFAConditionalDiagnostic
    torch.set_num_threads(4)
    assert flash_attn.flash_attn_func is fa.flash_attn_func
    assert digest(fa.__file__) == p['installed_FA_interface_sha256']
    artifact_path = Path(p['private_artifact_path'])
    assert artifact_path.stat().st_size == p['private_artifact_bytes']
    assert digest(artifact_path) == p['private_artifact_sha256']
    tick = time.perf_counter()
    private = torch.load(artifact_path, map_location='cpu', weights_only=True)
    r['CPU_load_seconds'] = time.perf_counter() - tick
    saved = private['layers']['19']['coeff']
    meta = saved['FA_layout']
    B, H, T, D = saved['mcontent'].shape
    KH, G = meta['kv_heads'], meta['groups']
    assert (B, H, KH, T, D) == (1, 16, 4, 351, 256) and H == KH * G
    assert meta['mapping'] == 'query_head // groups = compact_kv_head'
    paired = {name: private['layers']['19']['paired']['c'][name]
              for name in ('query', 'key', 'value', 'attention_output')}
    actual = {step: {name: private['scoring'][step]['19']['c'][name]
                     for name in ('query', 'key')} for step in ('0', *p['fixed_steps'])}
    seed = saved['mcontent']
    saved_coeff = saved['FA_coeff']
    r['input'] = source['input']
    r['saved_B1_vs_B2_endpoint_drift'] = hybrid['saved_B1_vs_B2_endpoint_drift']
    del private, saved
    gc.collect()
    with np.load(p['hybrid_vectors_path'], allow_pickle=False) as archive:
        reference = {step: {'R0': torch.from_numpy(archive[step + '_R0'].copy()),
                            'qk_prediction': torch.from_numpy(archive[step + '_qk_prediction'].copy())}
                     for step in p['fixed_steps']}

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    native = {name: value.transpose(1, 2).to('cuda').contiguous()
              for name, value in paired.items() if name != 'attention_output'}
    r['public_B2_GPU_layout'] = {name: layout(value) for name, value in native.items()}
    assert all(value.dtype == torch.bfloat16 and value.is_cuda and value.is_contiguous()
               for value in native.values())
    kwargs = dict(p['native_FA_kwargs'])
    kwargs['window_size'] = tuple(kwargs['window_size'])
    assert kwargs['dropout_p'] == 0 and kwargs['return_attn_probs'] is True
    r['status'] = 'one_public_B2_LSE_call'
    save()
    torch.cuda.synchronize()
    tick = time.perf_counter()
    r['native_FA_entered'] += 1
    with torch.no_grad():
        output, lse, unused = fa.flash_attn_func(native['query'], native['key'], native['value'], **kwargs)
    torch.cuda.synchronize()
    r['native_FA_returned'] += 1
    r['native_FA_seconds'] = time.perf_counter() - tick
    assert unused is None or unused.numel() == 0, 'Refuse a quadratic attention-probability buffer.'
    assert output.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert lse.shape == (2, H, T) and bool(torch.isfinite(lse).all())
    r['public_B2_replay_drift'] = drift(cpu(output), paired['attention_output'])
    operands = {'q0': native['query'][0:1].transpose(1, 2), 'q1': native['query'][1:2].transpose(1, 2),
                'k0': native['key'][0:1].transpose(1, 2), 'k1': native['key'][1:2].transpose(1, 2),
                'v0': native['value'][0:1].transpose(1, 2), 'u': seed.to('cuda').contiguous(),
                'lse0': lse[0:1], 'lse1': lse[1:2]}
    actual_gpu = {step: {name: value.to('cuda').contiguous() for name, value in entry.items()}
                  for step, entry in actual.items()}
    r['actual_GPU_layouts'] = {step: {name: layout(value) for name, value in entry.items()}
                              for step, entry in actual_gpu.items()}
    lengths = RightPaddedLengths([T], T, operands['q0'].device)
    op = VendorFAConditionalDiagnostic(library, build['library']['sha256'])
    replay_artifact = {'public_B2_LSE': cpu(lse), 'points': {}}
    del output, unused
    first_coeff = None
    arrays = {}
    P = source['input']['prompt_length']
    keep = set(source['input']['keep'])
    r['status'] = 'three_frozen_conditional_diagnostics'
    for step in p['fixed_steps']:
        tick = time.perf_counter()
        result = op(operands, {'qc': actual_gpu['0']['query'], 'kc': actual_gpu['0']['key'],
                              'qa': actual_gpu[step]['query'], 'ka': actual_gpu[step]['key']},
                    p['native_FA_kwargs']['softmax_scale'], lengths)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        result = {name: cpu(value) for name, value in result.items()}
        replay = {name: result[name] for name in ('dq', 'dk', 'dv')}
        if first_coeff is None:
            first_coeff = replay
        independent = {name: bool(torch.equal(value, first_coeff[name])) for name, value in replay.items()}
        assert all(independent.values()), 'Actual diagnostic operands changed the frozen coefficient path.'
        dq = actual['0']['query'].double() - actual[step]['query'].double()
        dk = actual['0']['key'].double() - actual[step]['key'].double()
        d_saved = prediction(saved_coeff, dq, dk)
        d_replay = prediction(replay, dq, dk)
        assert float((d_saved - reference[step]['qk_prediction']).abs().max()) < 1e-7
        tfp = result['conditional_fp32'].double().sum(1)
        tbf = result['conditional_bf16'].double().sum(1)
        r0 = reference[step]['R0'].double()
        fields = {'saved_qk_prediction': d_saved, 'replayed_qk_prediction': d_replay,
            'T_FP32': tfp, 'T_BF16': tbf, 'R0': r0,
            'coefficient_replay_difference': d_saved - d_replay,
            'QK_interaction_and_multiplier_rounding': d_replay - tbf,
            'tile_weight_BF16_cast': tbf - tfp,
            'finite_softmax_condition_and_native_precision': tfp - r0}
        keys = ('coefficient_replay_difference', 'QK_interaction_and_multiplier_rounding',
                'tile_weight_BF16_cast', 'finite_softmax_condition_and_native_precision')
        closure = sum(fields[name] for name in keys) - (d_saved - r0)
        assert float(closure.abs().max()) < 1e-7 and abs(float(closure.sum())) < 1e-7
        original = hybrid['points'][step]['fields']['routing_at_baseline_values_prediction_error']['net']
        assert abs(float((d_saved - r0).sum()) - original) < 1e-7
        deleted = set(source['points'][step]['input_receipt']['deleted_positions'])
        assert deleted <= keep
        groups = {'deleted': sorted(deleted), 'kept': sorted(keep - deleted),
                  'other_prompt': sorted(set(range(P)) - keep), 'response': list(range(P, T))}
        assert sum(map(len, groups.values())) == T
        if step == '20':
            assert deleted == keep, 'The allEOS guard must delete every original eligible input.'
        row = {'input_receipt': source['points'][step]['input_receipt'],
            'diagnostic_seconds': elapsed, 'coefficient_independent_of_actual_A': independent,
            'coefficient_replay_drift': {name: drift(replay[name], saved_coeff[name]) for name in replay},
            'original_route_prediction_error': original,
            'fields': {name: stats(value) for name, value in fields.items()},
            'groups': {group: {'count': len(indices), 'fields': {name: stats(value[0, indices])
                       for name, value in fields.items()}} for group, indices in groups.items()},
            'max_token_algebra_closure': float(closure.abs().max()), 'total_algebra_closure': float(closure.sum())}
        for name, value in fields.items():
            assert abs(sum(group['fields'][name]['net'] for group in row['groups'].values()) - float(value.sum())) < 1e-7
            arrays[step + '_' + name] = value.numpy()
        r['points'][step] = row
        replay_artifact['points'][step] = result
        save()
    assert op.calls_entered == op.calls_returned == 3
    assert r['native_FA_entered'] == r['native_FA_returned'] == 1
    torch.save(replay_artifact, A / 'diagnostic_coefficients_private.pt')
    np.savez_compressed(A / 'conditional_signed_rows.npz', **arrays)
    r['artifacts'] = {name: {'sha256': digest(A / name), 'bytes': (A / name).stat().st_size}
                      for name in ('diagnostic_coefficients_private.pt', 'conditional_signed_rows.npz')}
    assert digest(artifact_path) == p['private_artifact_sha256']
    assert digest(fa.__file__) == p['installed_FA_interface_sha256']
    for item in p['protected_sources']:
        assert digest(item['path']) == item['sha256'], item['path']
    r['status'] = 'three_FA19_conditional_score_contractions_complete'
except Exception:
    r['status'] = 'failed'
    r['error'] = traceback.format_exc()
finally:
    signal.alarm(0)
    r['seconds'] = time.perf_counter() - started
    r['diagnostic_finite_entered'] = op.calls_entered if op is not None else 0
    r['diagnostic_finite_returned'] = op.calls_returned if op is not None else 0
    r['diagnostic_phase_launches_expected'] = 3 * r['diagnostic_finite_entered']
    if torch is not None and torch.cuda.is_initialized():
        r['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        r['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    r['scope'] = 'Saved actual single-layer operator coordinates; no new method, full-model counterfactual, RISE or MAS. Three-term semantic attribution is not inferred from algebraic closure alone.'
    r['precision_scope'] = 'T uses the pinned extension FP32 score MMA and row reductions with BF16 operands; W_FP32 and its actual original BF16 conversion are separate. QK term also contains midpoint/GEMM/output rounding. Softmax term also contains BF16 upstream seed, B1/B2 endpoint transfer and native output precision; it is not a pure real-arithmetic slope measurement.'
    r['group_contract'] = 'T/R0 use output-query locations; Q/K predictions use their operand locations. Same index can represent different roles. Signed token-net groups are coordinate ledgers, not original-source causal effects.'
    r['allEOS_scope'] = 'Step20 is the original real B1 all-eligible-deleted input, not the paired B2 EOS endpoint. Its known B1/B2 drift is preserved; zero residual is not imposed.'
    save()
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in [*p['files_sha256'], 'protocol.json', 'build_results.json', 'compile.log',
                     'results.json', 'conditional_signed_rows.npz']:
            if (A / name).is_file():
                bundle.write(A / name, name)
    print(json.dumps({'status': r['status'], 'seconds': r['seconds'],
        'native_FA_entered': r['native_FA_entered'], 'native_FA_returned': r['native_FA_returned'],
        'diagnostic_finite_entered': r['diagnostic_finite_entered'],
        'diagnostic_finite_returned': r['diagnostic_finite_returned'], 'error': r.get('error')}), flush=True)
