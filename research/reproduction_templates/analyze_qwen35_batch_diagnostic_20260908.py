"""Recompute bounded diagnostics from captured native arrays; no model calls."""
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
R = A.parent / 'DeltaTrace'
D = A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_batch_diagnostic_20260908_v1'
sha = lambda b: hashlib.sha256(b).hexdigest()
sys.path.insert(0, str(R / 'research/runtime'))
from native_batch_diagnostics import metrics

bundle = D / 'review_bundle.zip'
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).name == name
        (D / name).write_bytes(z.read(name))
raw = (D / 'results.json').read_bytes()
r = json.loads(raw)
p = json.loads((D / 'protocol.json').read_bytes())
assert p == r['protocol']
assert sha((D / 'study.py').read_bytes()) == p['study_sha256']
assert (D / 'study.py').read_bytes() == (A / 'qwen35_batch_diagnostic_20260908.py').read_bytes()
for name, key in [('native_batch_diagnostics.py','diagnostic_runtime_sha256'),
                  ('official_fixed_text_inputs.py','input_runtime_sha256')]:
    assert sha((D / name).read_bytes()) == p[key]
assert r['status'] == 'bounded_diagnostics_complete_original_gate_reported_separately', r.get('error')
assert r['model_load_attempts'] == r['model_loads'] == 1
assert r['root_forward_attempts'] == r['root_forwards_entered'] == r['root_forwards_completed'] == 2
assert r['auxiliary_FA_attempts'] == r['auxiliary_FA_completed'] == 1
assert r['CPU_reference_attempts'] == r['CPU_reference_completed'] == 1
assert r['attributions'] == r['quality_queries'] == r['generation_calls'] == 0
assert r['sources_before'] == r['sources_after']
assert r['checkpoint_stats_before'] == r['checkpoint_stats_after']
parent_raw = (A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v3/results.json').read_bytes()
assert sha(parent_raw) == p['required_parent_raw_sha256']
parent = json.loads(parent_raw)
assert r['input_metadata'] == parent['input_metadata']
assert r['calls'][0]['target_logprobs'] == parent['calls'][0]['target_logprobs']
assert r['calls'][1]['target_logprobs'] == parent['calls'][2]['target_logprobs']
arrays = {}
for item in r['diagnostics']['artifacts']:
    path = D / item['file']
    assert sha(path.read_bytes()) == item['sha256']
    with np.load(path, allow_pickle=False) as f:
        arrays[path.stem] = {k: f[k] for k in f.files}
    assert list(arrays[path.stem]) == item['keys']


def check(a, b, saved):
    result = metrics(a, b)
    for key, value in result.items():
        if isinstance(value, float):
            assert np.isclose(value, saved[key], rtol=1e-12, atol=1e-15), (key,value,saved[key])
        else:
            assert value == saved[key]
    return result


target = check(r['calls'][0]['target_logprobs'][0], r['calls'][1]['target_logprobs'][0], r['batch_target_comparison'])
fla_batch = {k:check(arrays['FLA_prefix_call0'][k], arrays['FLA_prefix_call1'][k], r['FLA_batch_prefix'][k])
             for k in ['q','k','v','g','beta','output']}
fla_ref = check(arrays['FLA_prefix_call0']['output'], arrays['FLA_CPU_reference']['output'], r['CPU_reference'])
fa = check(arrays['FA_dense_actual_call0']['output'], arrays['FA_varlen_same_QKV']['output'], r['same_QKV_FA_comparison'])
prefix_module = {k:metrics(v, arrays['native_module_prefixes_call1'][k])
                 for k,v in arrays['native_module_prefixes_call0'].items()}
modules = r['diagnostics']['module_comparisons']
assert len(modules) == 41 and len({x['name'] for x in modules}) == 41
receipts = [{x['name']:x for x in call} for call in r['diagnostics']['module_receipts']]
for item in modules:
    name = item['name']
    assert receipts[0][name]['shape'] == receipts[1][name]['shape'] == item['shape']
    assert (receipts[0][name]['sha256_float32_values'] == receipts[1][name]['sha256_float32_values']) == (item['changed_elements'] == 0)
for call in r['calls']:
    c=call['python_dispatch']
    assert call['status']=='complete' and call['layer_calls']==list(range(32))
    assert c.get('FLA_chunk',0)==c.get('causal_conv',0)==24
    assert c.get('FA_dense',0)+c.get('FA_varlen',0)==8
    assert sum(c.get(k,0) for k in ['FLA_recurrent','torch_chunk_fallback','torch_recurrent_fallback'])==0
path = D / r['auxiliary_FA_profile']['file']
assert sha(path.read_bytes()) == r['auxiliary_FA_profile']['sha256']
events = json.loads(path.read_bytes())['traceEvents']
kernels = Counter(x['name'] for x in events if x.get('cat')=='kernel')
assert kernels, 'No actual GPU kernel trace'
fa_kernels = {k:v for k,v in kernels.items() if 'flash_fwd' in k}
assert fa_kernels, 'Native FA call lacks GPU evidence'
screen = r['original_batch_screen']
assert screen['relative_passed'] == (target['relative_L2'] <= p['relative_L2_limit'])
assert screen['absolute_passed'] == (target['max_abs'] <= p['max_abs_logprob_limit'])
boundaries = {}
for lo,hi in [(0,64),(64,128),(128,129)]:
    boundaries[f'{lo}:{hi}'] = metrics(arrays['FLA_prefix_call0']['output'][:,lo:hi],
                                      arrays['FLA_CPU_reference']['output'][:,lo:hi])
summary = {
    'status':'bounded_native_diagnostics_verified', 'raw_sha256':sha(raw),
    'study_sha256':p['study_sha256'], 'protocol_sha256':sha((D/'protocol.json').read_bytes()),
    'bundle_sha256':sha(bundle.read_bytes()), 'parent_raw_sha256':p['required_parent_raw_sha256'],
    'budget':{'model_loads':1,'native_forwards':2,'auxiliary_FA':1,'official_CPU_reference':1,
              'attributions':0,'quality_queries':0,'generation_calls':0},
    'sources':r['sources_after'], 'checkpoint_stats_unchanged':True,
    'all_three_target_vectors_equal_previous_native_preflight':True,
    'target_B1_B2':target, 'original_screen':screen,
    'native_module_comparisons':modules, 'persisted_module_prefix_comparisons':prefix_module,
    'FLA_B1_B2_prefix':fla_batch, 'FLA_vs_official_CPU_reference':fla_ref,
    'FLA_vs_official_CPU_by_chunk':boundaries, 'same_QKV_FA_dense_varlen':fa,
    'FA_capture_metadata':next(x['metadata'] for x in r['diagnostics']['artifacts'] if x['file']=='FA_dense_actual_call0.npz'),
    'auxiliary_actual_FA_kernels':fa_kernels, 'selected_state_schedule':r['selected_state_schedule'],
    'dispatch_per_forward':[x['python_dispatch'] for x in r['calls']],
    'diagnostic_cost_not_steady_performance':[{'seconds':x['diagnostic_seconds'],
        'peak_allocated_bytes':x['diagnostic_peak_allocated_bytes']} for x in r['calls']],
    'CPU_reference_seconds':r['CPU_reference']['seconds'], 'job_seconds':r['job_seconds'],
    'artifacts':r['diagnostics']['artifacts'],
    'limits':['Only NI0 is paired across B1/B2; MH1 is the padding/batch companion.',
        'Layer0 FLA reference covers first129 actual tokens only, not all layers or backward.',
        'FA comparison preserves values/dtype but replay tensors are contiguous; no model outputs are replaced.',
        'Full module metrics came from inspected runtime; selected129-token prefix arrays are independently recomputed here.',
        'No original quality curves, attribution support, deletion sign or steady speed claim.']}
(A/'qwen35_batch_diagnostic_summary_20260908.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','raw_sha256','target_B1_B2','original_screen',
                                      'FLA_B1_B2_prefix','FLA_vs_official_CPU_reference',
                                      'same_QKV_FA_dense_varlen','selected_state_schedule']}))
print(json.dumps({'module_relative_L2':{x['name']:x['relative_L2'] for x in modules}}))
