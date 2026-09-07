"""Recompute both batch comparisons and real kernel evidence, including failed gates."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
A = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
jobs = []
for version in [1, 2, 3]:
    D = A / f'snapshot${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v{version}'
    raw = (D / 'results.json').read_bytes(); d = json.loads(raw)
    assert sha((D / 'study.py').read_bytes()) == d['protocol']['study_sha256']
    assert json.loads((D / 'protocol.json').read_text()) == d['protocol']
    assert sha((D / 'official_fixed_text_inputs.py').read_bytes()) == d['protocol']['input_runtime_sha256']
    jobs.append({'directory': D, 'data': d, 'raw_sha256': sha(raw)})
assert all(j['data']['status'] == 'failed' for j in jobs)
assert [j['data']['root_forward_attempts'] for j in jobs] == [0, 1, 3]
assert [j['data']['root_forwards_completed'] for j in jobs] == [0, 0, 3]
assert all(j['data']['model_loads'] == 1 for j in jobs)
assert 'wordfreq' in jobs[0]['data']['error']
assert 'Required: 76288, Hardware limit: 65536' in jobs[1]['data']['error']
d = jobs[2]['data']; D = jobs[2]['directory']
assert 'relative_L2_limit' in d['error'] and 'CPU_reference' not in d
assert not (D / 'first_layer_prefix_reference.npz').exists()
assert d['scheduling_change']['function_bodies_identical']
assert d['sources_during']['explicit_changes'] == ['fla/utils.py', 'fla/ops/common/chunk_delta_h.py']
for j, row in enumerate(d['calls']):
    assert row['status'] == 'complete' and row['layer_calls'] == list(range(32))
    expected = {'causal_conv': 24, 'FLA_chunk': 24, 'FA_dense' if j < 2 else 'FA_varlen': 8}
    assert row['python_dispatch'] == expected
    assert row['input_shape'] == [[1, 605], [1, 368], [2, 605]][j]
for case in d['input_metadata']:
    assert case['user_token_sequence_exact'] and case['author_positions_equal_offsets']
    assert not case['boundary_crossing_positions']
    assert case['replacement_token_id'] == 248046 and not case['gold_and_cached_generation_indices_remapped']
comparisons = []
for j in [0, 1]:
    a = np.asarray(d['calls'][j]['target_logprobs'][0], dtype=np.float64)
    b = np.asarray(d['calls'][2]['target_logprobs'][j], dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error = np.abs(a-b)
    comparisons.append({'case': d['protocol']['selection'][j], 'tokens': len(a),
        'relative_L2': float(np.linalg.norm(a-b)/np.linalg.norm(a)),
        'max_abs_logprob_difference': float(error.max()), 'mean_abs_logprob_difference': float(error.mean()),
        'absolute_difference_quantiles_50_90_99': np.quantile(error, [.5, .9, .99]).tolist(),
        'B1_logprob_sum': float(a.sum()), 'B2_logprob_sum': float(b.sum()),
        'original_relative_gate_passed': bool(np.linalg.norm(a-b)/np.linalg.norm(a) <= d['protocol']['relative_L2_limit']),
        'original_max_absolute_gate_passed': bool(error.max() <= d['protocol']['max_abs_logprob_limit'])})
profiles = []
for job in jobs[1:]:
    for i, row in enumerate(job['data']['calls']):
        if 'profile' not in row: continue
        file = job['directory'] / row['profile']['file']
        assert sha(file.read_bytes()) == row['profile']['sha256']
        with file.open() as f: trace = json.load(f)
        counts = Counter(); durations = Counter()
        for event in trace['traceEvents']:
            if event.get('cat') == 'kernel':
                counts[event['name']] += 1; durations[event['name']] += event['dur']
        record = {'job': job['directory'].name, 'call_index': i, 'original_profile_sha256': row['profile']['sha256'],
            'total_GPU_kernel_events': sum(counts.values()),
            'native_FA_primary_kernel_events': sum(v for k, v in counts.items() if 'flash_fwd_kernel<' in k or 'flash_fwd_splitkv_kernel<' in k),
            'native_FA_splitkv_kernel_events': sum(v for k, v in counts.items() if 'flash_fwd_splitkv_kernel<' in k),
            'native_FA_combine_kernel_events': sum(v for k, v in counts.items() if 'flash_fwd_splitkv_combine_kernel<' in k),
            'FLA_state_kernel_events': sum(v for k, v in counts.items() if 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64' in k),
            'named_kernel_counts': dict(counts), 'named_kernel_total_us': dict(durations),
            'scope': 'Actual warm/shape-initialization profile, including native autotuner trials. Kernel event counts are not model-call counts or steady latency.'}
        (A / ('qwen35_kernel_counts_' + job['directory'].name + '_' + str(i) + '.json')).write_text(json.dumps(record, indent=2))
        print(json.dumps({'profile_job':job['directory'].name,'call':i,'FA_primary':record['native_FA_primary_kernel_events'],'FLA_state':record['FLA_state_kernel_events'],
            'matching_names':{k:v for k,v in counts.items() if 'flash' in k.lower() or 'blockdim64' in k}}), flush=True)
        if job is jobs[2]:
            assert record['native_FA_primary_kernel_events'] == 8 and record['FLA_state_kernel_events'] >= 24
        profiles.append(record); del trace
summary = {'status': 'actual_native_FA_FLA_execution_verified_batch_numerical_screen_failed',
    'raw_sha256': {str(i+1): j['raw_sha256'] for i, j in enumerate(jobs)},
    'actual_model_loads': 3, 'actual_forward_attempts_including_failure': 4, 'completed_model_forwards': 3,
    'FLA_chunk_API_calls_including_failure': 73, 'native_FA_API_calls':24,
    'CPU_reference_calls': 0, 'CPU_reference_skipped': 'An early batch assertion stopped before the planned diagnostic; prefix arrays were not persisted. Not a passing reference check.',
    'quality_queries': 0, 'generation_calls':0,'attributions':0,
    'batch_comparisons': comparisons, 'input_metadata': d['input_metadata'], 'profiles': profiles,
    'scheduling_change': d['scheduling_change'],
    'warm_or_shape_initialization_seconds': [row['elapsed_seconds_including_profile_if_enabled'] for row in d['calls']],
    'forward_peak_allocated_bytes': [row['peak_allocated_bytes'] for row in d['calls']],
    'job_seconds_including_failures': [j['data']['job_seconds'] for j in jobs],
    'whole_attribution_supported':False,'independent_quality_advantage':False,'batch_numerical_validation_complete':False,
    'interpretation': 'Default model actually executed through FA/FLA after a scheduling-only adaptation. A1% relative target-logprob screen failed at about2.45%; this is not itself proof of incorrect kernels or severe practical degradation. No post-hoc threshold relaxation.',
    'next': 'Keep the four-call budget closed. Persist layer/prefix evidence before diagnostic gates in the next separately frozen run; use native per-layer comparisons and the official CPU reference to locate BF16/GEMM versus FA/FLA/mask effects. Continue finite-recurrence derivation and gold remapping; no quality sweeps.'}
(A / 'qwen35_native_preflight_summary_20260908.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['profiles','input_metadata']},ensure_ascii=False))
