"""Verify full vectors, source provenance, complete cost and the preserved failed attempt."""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

A = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_dir = A / 'snapshot${ARTIFACT_ROOT}/codex_shared_fa_mlp_B4_20260908_v1'
failed2_dir = A / 'snapshot${ARTIFACT_ROOT}/codex_shared_fa_mlp_B4_20260908_v2'
F = A / 'snapshot${ARTIFACT_ROOT}/codex_shared_fa_mlp_B4_20260908_v3'
failed = json.loads((old_dir / 'results.json').read_text())
assert sha(old_dir / 'results.json') == '3ed932867ddee42135a118e0b80e34e2d767bf0deade8b4cfd02f548516e8600'
assert failed['status'] == 'failed' and failed['native_root_forwards'] == 1
assert failed['fresh_attributions'] == failed['finite_FA_calls_enqueued'] == 0
assert failed['extra_layer_replay_calls'] == failed['extra_native_fa_attention_calls'] == 1
assert sha(failed2_dir / 'results.json') == '1e62fc2935fb05acca0f4aab1be61b580f912b4e9a689105658e686cfdaebb58'
failed2 = json.loads((failed2_dir / 'results.json').read_text())
assert failed2['status'] == 'failed' and failed2['manual_attempts'] == 2
assert failed2['fresh_attributions'] == failed2['native_root_forwards'] == 1
assert failed2['extra_layer_replay_calls'] == failed2['extra_native_fa_attention_calls'] == failed2['finite_FA_calls_enqueued'] == 36
assert 'assert hooks() == 0' in failed2['error'] and 'UnboundLocalError' in failed2['error']
for directory in [old_dir, failed2_dir, F]:
    d = json.loads((directory / 'results.json').read_text())
    p = json.loads((directory / 'protocol.json').read_text())
    assert p == d['protocol'] and sha(directory / 'study.py') == p['study_sha256']
    for name, expected in p['sources'].items():
        assert sha(directory / name) == expected
assert d['status'] == 'shared_FA_native_MLP_B4_three_call_screen_complete'
assert d['fresh_attributions'] == d['native_root_forwards'] == d['manual_attempts'] == 3
assert d['extra_layer_replay_calls'] == d['extra_native_fa_attention_calls'] == d['finite_FA_calls_enqueued'] == 108
assert d['native_vjps'] == d['evaluation_forwards'] == d['ft_attribution_forwards'] == d['quality_queries'] == 0
assert d['native_sources_before'] == d['native_sources_after']
assert d['checkpoint_before'] == d['checkpoint_after']
assert sum(x['manual_attempts'] for x in [failed, failed2, d]) == 6
assert sum(x['native_root_forwards'] for x in [failed, failed2, d]) == 5
assert sum(x['fresh_attributions'] for x in [failed, failed2, d]) == 4
assert sum(x['extra_layer_replay_calls'] for x in [failed, failed2, d]) == 145
assert sum(x['finite_FA_calls_enqueued'] for x in [failed, failed2, d]) == 144
g = d['integration_groups'][0]
assert g['example_batch_size'] == 4 and g['selection'] == p['integration_groups'][0]
runs = g['runs']
assert len(runs) == 3
assert [(x['repeat'], x['mode']) for x in runs] == [(0, 'shared_mean_sac32'),
                                                  (1, 'shared_mean'), (1, 'shared_mean_sac32')]
for repeat in [1]:
    a, b = [next(x['result'] for x in runs if x['repeat'] == repeat and x['mode'] == mode)
            for mode in ['shared_mean', 'shared_mean_sac32']]
    assert a['signed_full_sequence'] == b['signed_full_sequence']
    assert a['endpoint_scores32'] == b['endpoint_scores32']
    reuse = b['native_MLP_reuse']
    assert reuse['saved_mm_outputs'] == reuse['reused_mm_outputs'] == 96
    assert reuse['retained_tensor_bytes'] == 0 and len(reuse['regions']) == 32
    assert reuse['installed_source_sha256'] == p['torch_checkpoint_source_sha256']
    assert all(x['input_exact'] and x['completed'] and x['save_calls'] == x['reuse_calls'] == 3 for x in reuse['regions'])
baseline_runs = [x['result'] for x in runs if x['mode'] == 'shared_mean' and not x['warm']]
candidate = next(x['result'] for x in runs if x['mode'] == 'shared_mean_sac32' and not x['warm'])
assert len(baseline_runs) == 1
assert baseline_runs[0]['signed_full_sequence'] == candidate['signed_full_sequence']
assert runs[0]['result']['signed_full_sequence'] == candidate['signed_full_sequence']
assert runs[0]['result']['endpoint_scores32'] == candidate['endpoint_scores32']
for row in runs:
    result = row['result']
    cost = result['end_to_end_cost']
    assert cost['measurement_entered']
    assert cost['native_forwards'] == 1 and cost['native_forward_trajectories'] == 8
    assert cost['extra_replay_calls'] == 36
    assert cost['public_FA_activity']['auxiliary_completed'] == 36
    assert len(cost['finite_FA_activity']) == 36
    for activity in cost['finite_FA_activity']:
        assert activity['calls_attempted'] == activity['calls_enqueued'] == 1
        assert activity['endpoint_mean'] == 'FP32_add_FP16_store_in_FA_shared_tile_from_existing_loads'
        assert activity['GQA_input_expansion'] is False
        assert activity['buffer_contract'][0]['shape'][0] == 4
    boundaries = result['native_layer_boundary_checks']
    if isinstance(boundaries, dict):
        boundaries = boundaries['paired_batch']
    assert all(x['native_input_exact'] and x['native_output_exact'] for x in boundaries)
profile = d['candidate_profile']
assert sha(F / profile['file']) == profile['sha256']
events = json.loads((F / profile['file']).read_text())['traceEvents']
names = Counter(x['name'] for x in events if x.get('cat') == 'kernel')
finite = sum(n for k, n in names.items() if 'deltatrace_fa_finite_p1_kernel' in k)
native = sum(n for k, n in names.items() if 'flash_fwd_kernel<' in k)
gemm = sum(n for k, n in names.items() if k.lower().startswith('mcblas') and 'gemm' in k.lower())
assert finite == native == 108 and gemm == 663
times = [x['all_in_seconds'] for x in baseline_runs]
new_time = candidate['all_in_seconds']
ratio = new_time / statistics.median(times)
summary = {'status': 'shared_FA_MLP_B4_three_calls_verified',
    'raw_sha256': sha(F / 'results.json'), 'protocol_sha256': sha(F / 'protocol.json'),
    'failed_parent_sha256': sha(old_dir / 'results.json'), 'second_failed_parent_sha256': sha(failed2_dir / 'results.json'), 'selection': g['selection'],
    'API_attempts_including_failed_attempts': 6, 'root_calls_including_failed_attempts': 5, 'completed_attributions': 4,
    'decoder_replays_including_failed_attempts': 145, 'auxiliary_FA_calls_including_failed_attempts': 145,
    'finite_operator_calls': 144, 'quality_queries': 0, 'FT_calls': 0,
    'measured_all_in_seconds': {'baseline_once': times, 'candidate_once': new_time},
    'candidate_to_baseline_ratio': ratio,
    'measured_peak_bytes': {'baseline_once': [x['peak_allocated_bytes'] for x in baseline_runs],
                            'candidate_once': candidate['peak_allocated_bytes']},
    'full_signed_vectors_and_endpoints_equal': True, 'sign_flips': 0,
    'saved_and_reused_native_mm_outputs_per_candidate_call': 96,
    'maximum_retained_cache_bytes': candidate['native_MLP_reuse']['maximum_retained_tensor_bytes'],
    'warm_profile': {'sha256': profile['sha256'], 'mcblas_gemm_count': gemm,
                     'finite_FA_kernels': finite, 'native_FA_kernels': native},
    'job_seconds': {'failed_binding': failed['job_seconds'], 'failed_hook_lifecycle': failed2['job_seconds'], 'corrected': d['job_seconds']},
    'automatic_expansion': False, 'stable_speedup_proven': False,
    'timing_limit': 'Only one measured baseline/candidate pair. All-in includes context construction/cleanup. Warm profiler excluded. No FT, quality or broader-length claim.',
    'next': 'Keep as a scoped candidate if faster; assess memory/throughput and obtain a bounded repeated comparison before default promotion. If no gain, stop expanding this cache setting.'}
(A / 'shared_fa_mlp_B4_summary_20260908.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
