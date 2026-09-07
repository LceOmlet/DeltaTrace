"""Recompute kernel times from raw traces; keep warm and end-to-end scopes distinct."""
import hashlib
import json
from collections import Counter
from pathlib import Path

A = Path(__file__).resolve().parent
F = A / 'snapshot${ARTIFACT_ROOT}/codex_fa_phase_profile_20260908_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
r = json.loads((F / 'results.json').read_text())
assert r['status'] == 'four_call_phase_profile_complete'
assert sha(F / 'study.py') == r['study_sha256'] == sha(A / 'fa_phase_profile_20260908.py')
assert len(r['operator_calls']) == 4
summary = {'status': 'phase_profiles_recomputed', 'raw_sha256': sha(F / 'results.json'),
           'operator_calls': 4, 'whole_attributions': 0, 'quality_queries': 0, 'phases': {},
           'numerics': r['numerics'], 'legacy_protocol_metadata_correction': r['protocol']['legacy_metadata_correction']}
for call in r['operator_calls']:
    if not call['profiled']:
        continue
    profile = call['profile']
    file = F / profile['file']
    assert sha(file) == profile['sha256']
    kernels = [e for e in json.loads(file.read_text())['traceEvents'] if e.get('cat') == 'kernel']
    assert [{'name': e['name'], 'dur_us': e['dur'], 'ts': e['ts']} for e in kernels] == profile['kernel_events']
    by_name = Counter()
    for e in kernels:
        by_name[e['name']] += e['dur']
    phase = {}
    for name, dur in by_name.items():
        if 'deltatrace_fa_finite_convert_dq' in name:
            phase['linear_conversion_us'] = dur
        elif 'deltatrace_fa_finite_p1_kernel' in name:
            for i in [0, 1, 2]:
                if '<' + str(i) + ',' in name:
                    phase['phase_' + str(i) + '_us'] = dur
    expected = 3
    assert len(phase) == expected and len(kernels) == 7
    phase.update(preparation_us=sum(by_name.values()) - sum(phase.values()),
                 total_kernel_us=sum(by_name.values()), kernel_count=len(kernels),
                 raw_profile_sha256=sha(file))
    summary['phases'][call['method']] = phase
old, new = summary['phases']['shared_mean'], summary['phases']['two_sweeps']
summary['combined_propagation_ratio'] = new['phase_2_us'] / (old['phase_1_us'] + old['phase_2_us'])
summary['conversion_fraction_of_new_kernels'] = new['linear_conversion_us'] / new['total_kernel_us']
summary['causal_limit'] = 'One profiled call per unchanged operator after warm-up. Locates regression inside merged propagation; does not separate atomics, register spills or transpose costs, and is not a new ordinary latency estimate.'

B = A / 'snapshot${ARTIFACT_ROOT}/codex_fa_shared_mean_integration_20260907_v1'
raw = json.loads((B / 'results.json').read_text())
assert sha(B / 'results.json') == '6c95d49f85e0eb74e3133d2476a09fce426cdcfe4a4262f8254d2ce3d81a2534'
profile = raw['shared_mean_B4_profile']
assert sha(B / profile['file']) == profile['sha256'] == '510897a4f641ac6f20f5421f18e11df6641794e04c0a2950a69cfef95999042a'
events = json.loads((B / profile['file']).read_text())['traceEvents']
kernels = [e for e in events if e.get('cat') == 'kernel']
durations, counts = Counter(), Counter()
for e in kernels:
    n = e['name'].lower()
    kind = ('finite_FA' if 'deltatrace_fa_finite_p1_kernel' in n else
            'native_FA' if 'flash_fwd_kernel<' in n else
            'named_mcblas_GEMM' if n.startswith('mcblas') and 'gemm' in n else 'other')
    durations[kind] += e['dur']
    counts[kind] += 1
assert counts['finite_FA'] == counts['native_FA'] == 108
total = sum(durations.values())
summary['existing_real_B4_warm_profile'] = {
    'raw_profile_sha256': profile['sha256'], 'raw_result_sha256': sha(B / 'results.json'),
    'selection': raw['integration_groups'][1]['selection'], 'GPU_kernel_count': len(kernels),
    'kernel_durations_us': dict(durations), 'kernel_counts': dict(counts),
    'duration_fractions': {k: v / total for k, v in durations.items()},
    'scope': 'Existing same-backend B4 warm attribution. May contain compiler warm kernels. Sum of GPU kernel durations is not steady end-to-end latency or a proven wall-time upper bound.',
    'decision': 'Repeated GEMMs are a larger measured component than finite FA here. Test the already justified last32-layer public-SAC MLP reuse on current FA and original B4; no claim that all FA optimization is exhausted.'}
(A / 'fa_phase_profile_summary_20260908.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
