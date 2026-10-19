"""Extend the existing original-storage selector to the verified 16384 limit.

The same native runner owns every calculation. Inputs above the accepted
storage limit keep its existing offloaded route through 32768.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('apply_small_gpu_resident_resource.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '8125b22b248cf200303d12b4bdfb449beefd1c54841378794d7fb102f542858a'
code = owner.read_text()
for old, new in (
    ('existing-DT-gpu-resident-pair-results.json', 'existing-DT-measured-source-resident-pair-v2-results.json'),
    ('2048 < row[\'paired_shape\'][1] <= 3072', "2048 < row['paired_shape'][1] <= 16384"),
    ("    assert candidate['peak_allocated'] < 55 * 2**30\n", ''),
    ('small-GPU-resident-resource-accepted-v1', 'validated-16384-GPU-resource-accepted-v2'),
    ('resource_extent_limit=2048', 'resource_extent_limit=16384'),
    ('paired.shape[1]<=2048', "paired.shape[1]<=state['resource_extent_limit']"),
    ("installed=state is None\n    if installed:\n        assert runner.attribute.__name__=='attribute'\n        native_attribute=runner.attribute",
     "assert state is not None and state['resource_extent_limit'] in (2048,16384)\n    previous_state=state\n    installed=state['resource_extent_limit']!=16384\n    if installed:\n        assert runner.attribute.__name__=='choose_original_mixer_storage'\n        native_attribute=state['original_attribute']"),
    ('original_attribute=native_attribute)',
     "original_attribute=native_attribute,\n            previous_observed={k:v for k,v in previous_state.items() if k!='original_attribute'})"),
    ('small_GPU_resident_resource_override', 'validated_GPU_resident_resource_override'),
    ('applied_original_storage_option_bounded_small_inputs', 'applied_original_storage_option_verified_16384'),
    ('completed_runtime_validated_GPU_resident_resource_override', 'completed_runtime_validated_16384_GPU_resource_override'),
    ('Current loaded runner only; long-input production route unchanged',
     'Current loaded runner only; original offloaded route retained above16384 through32768'),
    ('Eight simultaneous natural B4 pairs of extent2217..2401 fit below55GiB; production resident route restricted to extent<=2048. Original32768 offloaded route unchanged.',
     'Eight simultaneous original artificial B4/exact16384 capacity calls completed; native natural widths5134..9228 have identical DT vectors and unchanged compute counts. Resident route bounded to16384, larger contexts keep original32768 offloaded route.'),
):
    assert old in code, old
    code = code.replace(old, new)
anchor = "owner = Path(__file__).with_name('inspect_native_workers.py')"
assert code.count(anchor) == 1
code = code.replace(anchor, '''capacity = json.loads((AUDIT / 'resident-16384-native-capacity-results.json').read_bytes())
assert capacity['status'] == 'completed' and not capacity['failed_ranks']
assert sorted(capacity['completed_ranks']) == list(range(8))
for row in capacity['rows']:
    detail = row['value']
    assert detail['input_shape'] == [4,16384]
    assert detail['native_report']['actual_context_lengths'] == [16384]*4
    assert detail['native_report']['minibatch_size'] == 4
    assert all(detail['all_QVA_finite'].values()) and detail['RNG_unchanged']
    assert detail['original_options_restored'] and detail['original_method_restored']

''' + anchor)
exec(compile(code, str(owner), 'exec'), globals())
