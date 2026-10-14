"""Account saved native rank metadata; no model, tensor or training calls."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT / 'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
directory = OUT / 'live-native-rank-metadata-once-v76'
installation_path = directory / 'installation.json'
installation = json.loads(installation_path.read_bytes())
captures = []
for rank in (0, 1):
    path = directory / f'rank{rank}-native-update.json'
    value = json.loads(path.read_bytes())
    assert value['rank'] == rank and value['completed_iteration_before_update'] == 5
    assert value['original_update_restored'] and value['original_optimizer_step_restored']
    assert value['actor_SHA256'] == '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
    assert value['core_SHA256'] == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
    assert len(value['native_metrics']['actor/grad_norm']) == 4
    for norm in value['native_norm_objects']:
        assert norm['norm_class'] == 'torch.distributed.tensor.DTensor'
        assert norm['placement_classes'] == ['torch.distributed.tensor._ops._math_ops._NormPartial']
        assert norm['placement_state'] == [dict(reduce_op='sum', norm_type=2.0)]
        assert norm['original_return_object_unchanged']
    captures.append(dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), value=value))

partial = [row['value']['native_metrics']['actor/grad_norm'] for row in captures]
global_per_step = [math.hypot(a, b) for a, b in zip(*partial)]
result = dict(
    unix=time.time(), status='one_natural_iteration6_native_metadata_observation_completed_methods_restored',
    driver_pid=installation['driver_pid'], driver_birth=installation['driver_birth'],
    installation=dict(path=str(installation_path), SHA256=hashlib.sha256(installation_path.read_bytes()).hexdigest()),
    captures=captures, native_local_norms_by_rank=partial,
    native_local_norm_mean_by_rank=[sum(x) / len(x) for x in partial],
    materialized_global_norm_per_optimizer_step=global_per_step,
    materialized_global_norm_mean=sum(global_per_step) / len(global_per_step),
    calculation='For each of four original updates, sqrt(rank0_partial_norm**2 + rank1_partial_norm**2); then average.',
    official_actor_core_unchanged=True, original_methods_restored_both_ranks=True,
    model_backward_optimizer_scheduler_calls_added=0, tensor_arithmetic_collectives_added=0,
    training_math_changes=0, rootcause_repair_verified=False,
    limits=[
        'Saved Python float metrics are combined on CPU; no extra live DTensor arithmetic or collective was executed.',
        'This establishes reporting scope, not incorrect CUDA clipping or the cause of historical growth.',
        'Historical missing rank metrics cannot be recovered by assuming sqrt(2).',
    ],
)
path = directory / 'completed-accounting.json'
assert not path.exists(), 'Preserve prior completed observation receipt.'
path.write_text(json.dumps(result, indent=2) + '\n')
override = RUN / 'runtime-overrides/native-rank-metadata-once-v76-completed.json'
assert not override.exists()
override.write_text(json.dumps(result, indent=2) + '\n')
for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
    manifest_path = ROOT / name
    manifest = json.loads(manifest_path.read_bytes())
    def register(value):
        if isinstance(value, dict):
            if value.get('pid') == installation['driver_pid'] and value.get('method') == 'dt':
                value['completed_native_rank_metadata_observation'] = dict(
                    path=str(override), SHA256=hashlib.sha256(override.read_bytes()).hexdigest(),
                    completed_iteration=6, original_methods_restored=True,
                    training_math_changes=0, rootcause_repair_verified=False)
            else:
                for child in value.values():
                    register(child)
        elif isinstance(value, list):
            for child in value:
                register(child)
    register(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=result)))
