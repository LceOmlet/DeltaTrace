"""Record the existing native selector's verified 16384 runtime extension."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

owner = HERE / 'record_small_gpu_resource_deployment.py'
tree = ast.parse(owner.read_text())
code = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in node.targets))
for old, new in (
    ('actual-worker-imports-small-GPU-resident-resource-accepted-v1.json', 'actual-worker-imports-validated-16384-GPU-resource-accepted-v2.json'),
    ('small_GPU_resident_resource_override', 'validated_GPU_resident_resource_override'),
    ('applied_original_storage_option_bounded_small_inputs', 'applied_original_storage_option_verified_16384'),
    ("v['resource_extent_limit']==2048", "v['resource_extent_limit']==16384"),
    ("v['actor_scheduler_epoch']==21", "v['actor_scheduler_epoch']==36"),
    ('existing-DT-gpu-resident-pair-results.json', 'existing-DT-measured-source-resident-pair-v2-results.json'),
    ('small-GPU-resident-resource-override-completed.json', 'validated-16384-GPU-resource-override-completed.json'),
    ('completed_runtime_bounded_original_storage_selection', 'completed_runtime_verified_16384_original_storage_selection'),
    ('first_entire_production_iteration=22', 'first_entire_production_iteration=37'),
):
    assert old in code, old
    code = code.replace(old, new)
anchor = 'if target.exists():'
assert code.count(anchor) == 1
code = code.replace(anchor, '''deployment=json.loads((A/'validated-16384-GPU-resource-job.json').read_bytes())
capacity_path=A/'resident-16384-native-capacity-results.json'
capacity=json.loads(capacity_path.read_bytes())
assert capacity['status']=='completed' and sorted(capacity['completed_ranks'])==list(range(8))
record.update(adapter_commit=deployment['source_commit'],adapter_source=deployment['source'],
    adapter_source_SHA256=deployment['source_SHA256'],resource_extent_limit=16384,
    paired_reference='Warm original after candidate; cold original remains separately recorded',
    paired_median_seconds=[statistics.median(r['pair']['warm_baseline_wall_seconds'] for r in pair['same_input_pairs']),
                          statistics.median(r['pair']['candidate_wall_seconds'] for r in pair['same_input_pairs'])],
    capacity_verification=dict(path=str(capacity_path),SHA256=hashlib.sha256(capacity_path.read_bytes()).hexdigest()),
    capacity_scope='Eight simultaneous original B4/exact16384 completed; selector permits <=8 paired rows and <=16384 positions. Original32768 offloaded route is retained above this bound.',
    validation_calls=dict(DT=24,optimizer=0,rollout=0,iterations_excluded_from_uninstrumented_efficiency=[35,36]),
    candidate_GPU_Util_claimed=True,
    candidate_GPU_Util_scope='Original strict adjacent-eight-rank reducer: same-input candidate72.9047619%, warm original49.625%. This is validation-only, not a production iteration speedup.',
    production_efficiency_verified=False,queued_AppWorld_resident_policy_applied=False,
    arithmetic_owners_changed=False,persistence='Current TextCraft loaded runners; new16k production counters start at this completed override. AppWorld not yet applied.')
''' + anchor)
record = json.loads(run_python(code, timeout=40))
result = HERE.parents[2] / 'experiments/rl/results_dtpo_v9_negative2_validated_16384_resource_override_20261019_v4.json'
result.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=record['status'],local_result=str(result),
    latest_completed=record['latest_metric']['step'],first_full_production_iteration=37,
    resource_extent_limit=16384,production_efficiency_verified=False)))
