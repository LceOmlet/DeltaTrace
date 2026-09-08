"""Eleven frozen native FA operator calls on actual saved decoder19 operands.

No model, attribution, scorer, backward, explicit probabilities or new state
trajectory. Hybrid Q/K/V calls are operator contrasts, not model counterfactuals.
Captured native O_C/O_A remain the measured endpoints; replay drift and the
existing finite backend's BF16 seed cast are reported separately.

Reuse the successful V2 eleven-call computation and H2D layout preparation.
Adapt current MH2 source schema and verify native arguments through pinned source;
early step3 replacing the previous MH1 step1; native precision is unchanged.
"""
import ast
import gc
import hashlib
import importlib.util
import json
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path


A = Path(__file__).resolve().parent
p = json.loads((A / 'protocol.json').read_bytes())
started = time.perf_counter()
r = {'status': 'starting', 'protocol': p, 'native_FA_calls_entered': 0,
     'native_FA_calls_returned': 0, 'model_calls': 0, 'DT_calls': 0,
     'scorer_calls': 0, 'FT_calls': 0, 'backward_calls': 0,
     'generation_calls': 0,
     'calls': [], 'replays': {}, 'points': {}}
torch = None


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save():
    temporary = A / 'results.partial'
    temporary.write_text(json.dumps(r, indent=2, allow_nan=False))
    temporary.replace(A / 'results.json')


def timeout(*_args):
    raise TimeoutError('Frozen 180-second native FA operator budget exceeded.')


def nbytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    return 0


def signed_stats(value):
    value = value.double()
    assert bool(torch.isfinite(value).all())
    return {'net': float(value.sum()), 'positive_sum': float(value.clamp_min(0).sum()),
            'negative_sum': float(value.clamp_max(0).sum()), 'absolute_sum': float(value.abs().sum()),
            'positive_token_count': int((value > 0).sum()), 'negative_token_count': int((value < 0).sum())}


def project(seed, value):
    assert seed.device.type == value.device.type == 'cpu' and seed.shape == value.shape
    return (seed.double() * value.double()).flatten(2).sum(-1)


def tensor_layout(value):
    return {'shape': list(value.shape), 'stride': list(value.stride()),
            'dtype': str(value.dtype), 'device': str(value.device),
            'is_cuda': value.is_cuda, 'contiguous': value.is_contiguous()}


def native_call(label, operands):
    index = r['native_FA_calls_entered']
    assert index < 11 and label == p['native_call_schedule'][index]
    q, k, v = operands
    assert q.shape == (1, T, H, D) and k.shape == v.shape == (1, T, KH, D)
    assert all(x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() for x in operands)
    torch.cuda.synchronize()
    item = {'label': label, 'status': 'entered', 'q_shape': list(q.shape), 'k_shape': list(k.shape),
            'v_shape': list(v.shape), 'kwargs': p['native_FA_kwargs']}
    r['calls'].append(item)
    r['native_FA_calls_entered'] += 1
    tick = time.perf_counter()
    try:
        kwargs = dict(p['native_FA_kwargs'])
        kwargs['window_size'] = tuple(kwargs['window_size'])
        with torch.no_grad():
            output = fa.flash_attn_func(q, k, v, **kwargs)
        torch.cuda.synchronize()
        assert isinstance(output, torch.Tensor) and output.shape == q.shape and output.dtype == torch.bfloat16
        r['native_FA_calls_returned'] += 1
        item['status'] = 'returned'
        item['native_synchronized_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        captured = output.detach().to('cpu', copy=True)
        item['output_CPU_copy_seconds'] = time.perf_counter() - tick
        assert bool(torch.isfinite(captured).all())
        return captured
    except Exception:
        item['status'] = 'failed'
        raise
    finally:
        item['call_and_copy_seconds'] = time.perf_counter() - tick if 'native_synchronized_seconds' not in item else item['native_synchronized_seconds'] + time.perf_counter() - tick


try:
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(p['budget']['wall_time_seconds'])
    assert p['budget']['native_FA_calls'] == 11 and p['fixed_steps'] == ['3', '10', '20']
    for name, want in p['files_sha256'].items():
        assert digest(A / name) == want, name
    for item in p['successful_template_provenance']['files']:
        assert digest(Path(item['remote_path'])) == item['sha256']
    source_path = Path(p['source_results_path'])
    assert digest(source_path) == p['source_results_sha256']
    source = json.loads(source_path.read_bytes())
    assert source['status'] == 'MH2_FA19_GDN1_1native10replay2finite_internal_complete'
    assert digest(Path(p['source_protocol_path'])) == p['source_protocol_sha256']
    artifact_path = Path(p['private_artifact_path'])
    assert artifact_path.stat().st_size == p['private_artifact_bytes']
    r['private_artifact_sha256_before'] = digest(artifact_path)
    assert r['private_artifact_sha256_before'] == p['private_artifact_sha256']
    assert source['private_artifact']['sha256'] == p['private_artifact_sha256']
    assert source['input']['input_sha256'] == p['input_sha256']
    boundary_path = Path(p['source_boundary_results_path'])
    assert digest(boundary_path) == p['source_boundary_results_sha256']
    boundary = json.loads(boundary_path.read_bytes())
    assert source['input'] == boundary['cases']['morehopqa_2']['input']
    assert p['frozen_input_receipts'] == {step:boundary['cases']['morehopqa_2']['points'][step]['input_receipt'] for step in ('0',*p['fixed_steps'])}
    for point in source['layers']['19']['points'].values():
        assert point['mixer_calls']=={'module':1,'interface':1,'native_varlen':0,'native_dense':1}
    assert source['layers']['19']['actual_mask_shape'] is None
    for item in p['argument_provenance']['sources']:
        assert digest(Path(item['path']))==item['sha256']
    r['actual_native_argument_source']=p['argument_provenance']



    import torch
    import flash_attn
    import flash_attn.flash_attn_interface as fa
    import decoder19_conditional_decomposition_20260908 as ledger
    torch.set_num_threads(4)
    assert flash_attn.flash_attn_func is fa.flash_attn_func
    r['FA_interface_path'] = fa.__file__
    r['FA_interface_sha256_before'] = digest(Path(fa.__file__))
    assert r['FA_interface_sha256_before'] == p['installed_FA_interface_sha256']
    spec = importlib.util.find_spec('transformers.models.qwen3_5.modeling_qwen3_5')
    assert spec is not None and spec.origin
    model_source_path = Path(spec.origin)
    assert digest(model_source_path) == p['native_model_sha256']
    tree = ast.parse(model_source_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Qwen3_5Attention')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    scaling = [node.value for node in ast.walk(init) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self'
                       and t.attr == 'scaling' for t in node.targets)]
    assert len(scaling) == 1 and ast.dump(scaling[0]) == ast.dump(ast.parse('self.head_dim ** -0.5', mode='eval').body)
    config_path = Path(p['checkpoint_config_path'])
    assert digest(config_path) == p['checkpoint_config_sha256']
    config = json.loads(config_path.read_bytes())
    text_config = config.get('text_config', config)
    assert text_config['head_dim'] == 256 and p['native_FA_kwargs']['softmax_scale'] == 256 ** -.5
    r['scaling_source'] = {'model_source_path': str(model_source_path), 'model_sha256': p['native_model_sha256'],
        'verified_expression': 'self.head_dim ** -0.5', 'config_head_dim': 256,
        'native_scale': p['native_FA_kwargs']['softmax_scale'],
        'scope': 'Native scale is established from the pinned actual model/config and HF argument forwarding; source observer already asserted no windows, softcap or bias. Scalar dense kwargs were not persisted by this source. Four public FA replays independently quantify disagreement.'}

    tick = time.perf_counter()
    all_private = torch.load(artifact_path, map_location='cpu', weights_only=True, mmap=True)
    private = all_private['19']
    kw=private['official_B2_kwargs']
    assert kw['attention_mask'] is None and kw['past_key_values'] is None and kw['use_cache'] is False
    assert not set(kw).intersection({'scaling','dropout','sliding_window','softcap','is_causal','s_aux','alibi_slopes','window_size'})
    r['CPU_load_seconds'] = time.perf_counter() - tick
    r['private_tensor_CPU_bytes_loaded'] = nbytes(private)
    layer = {'coeff':private['coeff'], 'paired':private['native']['B2']}
    coeff = layer['coeff']
    paired_value0 = layer['paired']['c']['value'][0:1].clone()
    points = {step: {'d': {k: private['native'][step]['d'][k] for k in ledger.D_FIELDS},
                     'c': {k: private['native'][step]['c'][k] for k in ledger.C_FIELDS}}
              for step in ('0', *p['fixed_steps'])}
    meta = coeff['FA_layout']
    B, H, T, D = coeff['mcontent'].shape
    KH, G = meta['kv_heads'], meta['groups']
    assert B == 1 and D == 256 and H == KH * G and meta['query_heads'] == H
    assert meta['padded_length'] == T and meta['valid_lengths'] == [T]
    assert meta['mapping'] == 'query_head // groups = compact_kv_head'
    assert T == source['input']['total_length'] and H == text_config['num_attention_heads']
    assert KH == text_config['num_key_value_heads']
    for name in ('dq', 'dk', 'dv'):
        assert coeff['FA_coeff'][name].shape == (1, H, T, D)
    r['saved_B1_vs_B2_endpoint_drift'] = {}
    for label, step, paired_index in (('clean_vs_input', '0', 1), ('allEOS_vs_baseline', '20', 0)):
        r['saved_B1_vs_B2_endpoint_drift'][label] = {}
        for name in ('query', 'key', 'value', 'attention_output'):
            left = points[step]['c'][name].double()
            right = layer['paired']['c'][name][paired_index::2].double()
            assert left.shape == right.shape
            norm = float(right.norm())
            r['saved_B1_vs_B2_endpoint_drift'][label][name] = {
                'bitwise_equal_values': bool(torch.equal(left, right)),
                'relative_L2': float((left - right).norm()) / norm if norm else None,
                'max_absolute_difference': float((left - right).abs().max())}
    for step in p['fixed_steps']:
        actual_ledger = ledger.decompose(coeff, ledger.difference(points['0'], points[step]))
        frozen = source['layers']['19']['ledgers'][step]['internal']
        assert abs(actual_ledger['terms']['finite_FA_core_including_seed_cast'] - frozen['terms']['finite_FA_core_including_seed_cast']) < 1e-7
        r['points'][step] = {'input_receipt': p['frozen_input_receipts'][step],
            'original_FA_core_error': frozen['terms']['finite_FA_core_including_seed_cast'],
            'CPU_recomputed_original_FA_core_error': actual_ledger['terms']['finite_FA_core_including_seed_cast']}
    seed = coeff['mcontent'].transpose(1, 2).contiguous()
    seed_bf16 = seed.to(torch.bfloat16)
    dq = coeff['FA_coeff']['dq'].transpose(1, 2)
    dk = coeff['FA_coeff']['dk'].double().reshape(1, KH, G, T, D).sum(2).transpose(1, 2)
    dv = coeff['FA_coeff']['dv'].double().reshape(1, KH, G, T, D).sum(2).transpose(1, 2)
    actual_points = {step: {name: feature['c'][name] for name in ('query', 'key', 'value', 'attention_output')}
                     for step, feature in points.items()}
    r['FA_layout'] = meta
    r['selected_CPU_tensor_bytes'] = nbytes(actual_points) + nbytes(paired_value0) + sum(nbytes(x) for x in (seed, seed_bf16, dq, dk, dv))
    del private, all_private, layer, coeff, points, actual_ledger
    gc.collect()
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    gpu = {}
    for step, value in actual_points.items():
        gpu[step] = tuple(value[name].transpose(1, 2).to('cuda').contiguous() for name in ('query', 'key', 'value'))
    value0_gpu = paired_value0.transpose(1, 2).to('cuda').contiguous()
    r['prepared_GPU_operand_layouts'] = {
        step: {name: tensor_layout(value) for name, value in zip(('query', 'key', 'value'), operands)}
        for step, operands in gpu.items()}
    r['prepared_GPU_operand_layouts']['paired_EOS_value'] = tensor_layout(value0_gpu)
    save()
    assert value0_gpu.shape == (1, T, KH, D) and value0_gpu.dtype == torch.bfloat16
    r['status'] = 'four_actual_output_replays_then_seven_hybrids'
    save()
    for step in ('0', *p['fixed_steps']):
        replay = native_call('replay_' + step, gpu[step])
        original = actual_points[step]['attention_output']
        assert original.shape == replay.shape and original.dtype == torch.bfloat16
        change = replay.double() - original.double()
        denominator = float(original.double().norm())
        r['replays'][step] = {'bitwise_equal': bool(torch.equal(replay, original)),
            'relative_L2': float(change.norm()) / denominator if denominator else None,
            'max_absolute_difference': float(change.abs().max()),
            'stored_seed_projected_drift': signed_stats(project(seed, change)),
            'BF16_seed_projected_drift': signed_stats(project(seed_bf16, change))}
        save()
    xc0 = native_call('X_C0', (gpu['0'][0], gpu['0'][1], value0_gpu))
    oc = actual_points['0']['attention_output']
    P = source['input']['prompt_length']
    keep = set(source['input']['keep'])
    token_artifacts = {}
    for step in p['fixed_steps']:
        xa0 = native_call('X_A0_' + step, (gpu[step][0], gpu[step][1], value0_gpu))
        xca = native_call('X_CA_' + step, (gpu['0'][0], gpu['0'][1], gpu[step][2]))
        oa = actual_points[step]['attention_output']
        delta_q = (actual_points['0']['query'].double() - actual_points[step]['query'].double()).transpose(1, 2)
        delta_k = (actual_points['0']['key'].double() - actual_points[step]['key'].double()).transpose(1, 2)
        delta_v = (actual_points['0']['value'].double() - actual_points[step]['value'].double()).transpose(1, 2)
        qk_prediction = project(dq, delta_q) + project(dk, delta_k)
        v_prediction = project(dv, delta_v)
        actual_effect = project(seed, oc.double() - oa.double())
        R0 = project(seed, xc0.double() - xa0.double())
        VCVA = project(seed, oc.double() - xca.double())
        RA = project(seed, xca.double() - oa.double())
        terms = {'routing_at_baseline_values_prediction_error': qk_prediction - R0,
                 'content_at_clean_routing_prediction_error': v_prediction - VCVA,
                 'routing_content_interaction_contrast': R0 - RA}
        core = qk_prediction + v_prediction - actual_effect
        closure = sum(terms.values()) - core
        assert float(closure.abs().max()) < 1e-7 and abs(float(closure.sum())) < 1e-7
        assert abs(float(core.sum()) - r['points'][step]['original_FA_core_error']) < 1e-7
        cast_delta = project(seed_bf16.double() - seed.double(), oc.double() - oa.double())
        bf16_core = qk_prediction + v_prediction - project(seed_bf16, oc.double() - oa.double())
        assert float((core - bf16_core - cast_delta).abs().max()) < 1e-7
        fields = {'qk_prediction': qk_prediction, 'v_prediction': v_prediction,
                  'captured_actual_effect': actual_effect, 'R0': R0, 'VCVA': VCVA, 'RA': RA,
                  'core_error_at_stored_seed': core, 'core_error_at_BF16_seed': bf16_core,
                  'BF16_minus_stored_seed_actual_effect': cast_delta, **terms}
        row = r['points'][step]
        deleted = set(row['input_receipt']['deleted_positions'])
        assert deleted <= keep
        groups = {'deleted': sorted(deleted), 'kept': sorted(keep - deleted),
                  'other_prompt': sorted(set(range(P)) - keep), 'response': list(range(P, T))}
        assert sum(len(v) for v in groups.values()) == T
        row['fields'] = {name: signed_stats(value) for name, value in fields.items()}
        row['groups'] = {group: {'count': len(indices), 'fields': {name: signed_stats(value[0, indices])
                                      for name, value in fields.items()}} for group, indices in groups.items()}
        row['checks'] = {'max_token_three_term_closure': float(closure.abs().max()),
                         'total_three_term_closure': float(closure.sum()),
                         'total_RA_plus_VCVA_minus_captured_effect': float((RA + VCVA - actual_effect).sum()),
                         'core_minus_original_ledger': float(core.sum()) - row['original_FA_core_error']}
        for name, value in fields.items():
            assert abs(sum(group['fields'][name]['net'] for group in row['groups'].values()) - float(value.sum())) < 1e-7
            token_artifacts[step + '_' + name] = value.numpy()
        row['BF16_seed_contrast_differences'] = {
            'R0': signed_stats(project(seed_bf16.double() - seed.double(), xc0.double() - xa0.double())),
            'VCVA': signed_stats(project(seed_bf16.double() - seed.double(), oc.double() - xca.double())),
            'RA': signed_stats(project(seed_bf16.double() - seed.double(), xca.double() - oa.double()))}
        save()
    assert r['native_FA_calls_entered'] == r['native_FA_calls_returned'] == 11
    import numpy as np
    np.savez_compressed(A / 'signed_token_contrasts.npz', **token_artifacts)
    r['signed_token_contrasts'] = {'file': 'signed_token_contrasts.npz', 'sha256': digest(A / 'signed_token_contrasts.npz')}
    r['private_artifact_sha256_after'] = digest(artifact_path)
    assert r['private_artifact_sha256_after'] == r['private_artifact_sha256_before']
    assert digest(source_path) == p['source_results_sha256'] and digest(boundary_path) == p['source_boundary_results_sha256']
    r['FA_interface_sha256_after'] = digest(Path(fa.__file__))
    assert r['FA_interface_sha256_after'] == r['FA_interface_sha256_before']
    for item in p['argument_provenance']['sources']:
        assert digest(Path(item['path']))==item['sha256']
    assert digest(model_source_path) == p['native_model_sha256'] and digest(config_path) == p['checkpoint_config_sha256']
    for item in p['successful_template_provenance']['files']:
        assert digest(Path(item['remote_path'])) == item['sha256']
    r['status'] = 'MH2_eleven_native_FA_hybrid_contrasts_complete'
except Exception:
    r['status'] = 'failed'
    r['error'] = traceback.format_exc()
finally:
    signal.alarm(0)
    r['job_seconds'] = time.perf_counter() - started
    if torch is not None and torch.cuda.is_initialized():
        r['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        r['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    r['scope'] = 'Native FA hybrid operator contrasts on frozen saved decoder19 coordinates, not full-model counterfactuals, new attribution or new RISE/MAS.'
    r['term_interpretation'] = {
        'routing_at_baseline_values_prediction_error': 'Frozen q/k allocation minus native routing contrast at reference V0. Includes off-pair/B1-B2 transfer and quantization, not a pure kernel-error claim.',
        'content_at_clean_routing_prediction_error': 'Frozen v allocation minus actual clean-route content contrast; an observed difference does not establish a kernel bug.',
        'routing_content_interaction_contrast': 'Reference-value routing interaction, including native rounding; ideal expression m(P_C-P_A)(V0-V_A). Not an independent EOS causal contribution.',
        'seed_identity': 'E_stored = E_BF16 + <m_BF16-m_stored, O_C-O_A>; the seed term is not added a second time.'}
    r['precision_contract'] = 'Captured OC/OA remain actual endpoints. Native BF16 replay differences and the finite vendor BF16 seed conversion are reported. The three terms include default native finite-precision effects; algebraic closure is not a claim of exact real-arithmetic attention.'
    r['group_contract'] = 'Contrast terms group output-query locations; q/k/v predictions group their operand locations. Same token indices can label different roles. Positive/negative statistics are token-net sums, not independent source effects or headwise sign statistics.'
    save()
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in [*p['files_sha256'], 'protocol.json', 'results.json', 'signed_token_contrasts.npz']:
            if (A / name).is_file():
                bundle.write(A / name, name)
    print(json.dumps({'status': r['status'], 'native_FA_calls_entered': r['native_FA_calls_entered'],
                      'native_FA_calls_returned': r['native_FA_calls_returned'], 'seconds': r['job_seconds'],
                      'error': r.get('error')}), flush=True)
