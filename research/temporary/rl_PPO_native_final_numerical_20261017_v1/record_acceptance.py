"""Gather completed owner numerical, capacity and unchanged-performance receipts."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/PPO-native-final-numerical-20261017-v1'
OWNER = ROOT / 'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'


def read(path):
    return json.loads(Path(path).read_bytes())


def source(path):
    path = Path(path)
    return dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


numeric_path = AUDIT / 'warm-native-output-comparison.json'
numeric = read(numeric_path)
assert numeric['status'] == 'passed_warm_native_owner_output_criterion'
assert numeric['max_abs'] == 0 and numeric['different'] == 0 and numeric['shape'] == [8,8191]
raw_path = AUDIT / 'status.json'
raw = read(raw_path)
assert raw['status'] == 'completed_native_unobserved_update_output_replay'
assert all(item['full_preupdate_state_restore_exact'] and item['input_fields_exact'] for item in raw['completed'])
assert not list(AUDIT.glob('*optimizer-boundary-rank*.json')), 'No update observation hooks in the final comparison'
capacity_path = ROOT / 'receipts/PPO-32k-colocated-capacity-20261016-v1/execution3/complete-verification.json'
capacity = read(capacity_path)
assert capacity['status'] == 'completed_native_colocated_capacity'
assert (capacity['context'],capacity['response_length'],capacity['batch'],capacity['microbatch_per_rank']) == (32768,32767,8,4)
short_path = ROOT / 'receipts/PPO-32k-native-efficiency-20261016-v4/performance-exec3/status.json'
short = read(short_path)
assert short['status'] == 'completed' and short['native_short_dispatch'] == 'passed'
long_path = ROOT / 'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/status.json'
long = read(long_path)
assert long['status'] == 'completed' and long['exact_transport'] == 'passed'
long_comparison = next(item for item in long['comparisons'] if item['context'] == 32768)
assert short['comparison']['median_ratio'] <= 1 and long_comparison['median_ratio'] <= 1
bindings = {}
for worker in raw['workers']:
    for module, record in worker['sources'].items():
        assert source(record['path'])['SHA256'] == record['SHA256']
        bindings[module] = record
for worker in capacity['workers']:
    for module, record in worker['modules'].items():
        assert source(record['path'])['SHA256'] == record['SHA256']
        bindings[module] = record

result = dict(status='accepted_for_formal_AppWorld_PPO', collected=time.time(), owner_root=str(OWNER),
    upstream_commit='20bd331bdbc9026a5668e11362178e10ab7400c8', memory_patch_commit='1bb6520308d2ca0eea19b010922c9cfba3d8d130',
    numerical=dict(receipt=source(numeric_path), result=numeric,
        official_test='https://raw.githubusercontent.com/langfengQ/verl-agent/20bd331bdbc9026a5668e11362178e10ab7400c8/tests/utils/gpu_tests/test_activation_offload.py',
        comparison_scope='Same restored state/input after prior native current-process updates, as in owner warm-update sequence; no update read hooks',
        first_use_native_difference_retained=raw['comparisons'],
        first_use_difference_cause_claimed=False,
        universal_first_call_or_future_trajectory_bitwise_identity_claimed=False),
    capacity=dict(receipt=source(capacity_path),context=32768,response_axis=32767,batch=8,microbatch_per_rank=4,
                  scope='Native colocated actor/vLLM/critic; original oldlogprob/value/GAE/critic/actor updates, capacity fixture only'),
    performance=dict(short_receipt=source(short_path),long_receipt=source(long_path),
                     short_comparison=short['comparison'],long_comparison=long_comparison,
                     scope='Same source and original fast FA0 mode. 32k GPU-only baseline OOM, so long reference is official CPU saving; limited measured pairs, not full formal iteration speed'),
    actual_imported_sources=bindings, source_changed_since_final_numerical_and_full_capacity=False,
    older_long_performance_reused_for_existing_v4_verified_unchanged_long_branch=True,
    production_FA_deterministic=False, new_model_calls=0, checkpoint_create_calls=0,
    formal_training_started=False)
(AUDIT / 'formal-acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], owner_root=str(OWNER), numerical_max_abs=numeric['max_abs'],
    numerical_elements=numeric['total'], capacity=[32768,32767,8,4],
    short_ratio=short['comparison']['median_ratio'], long_ratio=long_comparison['median_ratio'])))
