"""Read the original81 target coefficients and pinned native loss contract.

The historical derivative carrier uses alpha=max(A[target]). Verify that this
is the actual target coefficient rather than assuming constant target credit.
No model, derivative, credit construction or optimizer is executed.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import dp_actor

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
captures = [json.loads((OUT / f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0, 1)]
prototype = DataProto.concat([DataProto.load_from_disk(item['input']) for item in captures])
carrier_path = OUT / 'historical-target-derivatives-v1/iteration81-target-derivative-carrier.pkl'
carrier = DataProto.load_from_disk(str(carrier_path))
assert len(prototype) == len(carrier) == 256
mask = carrier.batch['diagnostic_target_mask'].bool()
assert not (mask & ~prototype.batch['response_mask'].bool()).any()
original = prototype.batch['advantages']
diagnostic = carrier.batch['diagnostic_target_legacy']
alpha = original[mask].max()
fields_equal = {name: torch.equal(prototype.batch[name], carrier.batch[name]) for name in
    ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'response_mask', 'loss_mask')}
rows = []
for i in range(256):
    if not mask[i].any():
        continue
    actual = original[i][mask[i]]
    copied = diagnostic[i][mask[i]]
    rows.append(dict(row=i, native_reward=float(prototype.batch['token_level_scores'][i].sum()),
        target_tokens=int(mask[i].sum()), native_advantage_min=float(actual.min()),
        native_advantage_max=float(actual.max()), all_native_targets_equal_alpha=bool((actual == alpha).all()),
        max_abs_carrier_coefficient_difference=float((copied - actual).abs().max()),
        native_Q_min=float(prototype.batch['dt_q_estimates'][i][mask[i]].min()),
        native_Q_max=float(prototype.batch['dt_q_estimates'][i][mask[i]].max()),
        native_V_abs_max=float(prototype.batch['dt_v_estimates'][i][mask[i]].abs().max())))
actor_path = Path(inspect.getsourcefile(dp_actor))
core_path = Path(inspect.getsourcefile(core_algos))
assert hashlib.sha256(actor_path.read_bytes()).hexdigest() == '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert hashlib.sha256(core_path.read_bytes()).hexdigest() == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
meta = {key: prototype.meta_info.get(key) for key in ('global_steps', 'step', 'multi_turn', 'temperature', 'global_batch_info')}
report = dict(unix=time.time(), scope=__doc__, original_sources=[dict(path=item['input'],
    SHA256=hashlib.sha256(Path(item['input']).read_bytes()).hexdigest(), capture_keys=list(item)) for item in captures],
    original_meta_info=meta, carrier=dict(path=str(carrier_path), SHA256=hashlib.sha256(carrier_path.read_bytes()).hexdigest()),
    original81_input_fields_equal=fields_equal, native_alpha=float(alpha),
    target_policy_tokens=int(mask.sum()), target_rows=len(rows), rows=rows,
    all_native_target_values_equal_max_alpha=all(row['all_native_targets_equal_alpha'] for row in rows),
    max_abs_carrier_coefficient_difference=max(row['max_abs_carrier_coefficient_difference'] for row in rows),
    native_loss_owner=dict(actor_path=str(actor_path), actor_SHA256=hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        core_path=str(core_path), core_SHA256=hashlib.sha256(core_path.read_bytes()).hexdigest(),
        aggregation_source=inspect.getsource(core_algos.agg_loss),
        actor_update_source=inspect.getsource(dp_actor.DataParallelPPOActor.update_policy)),
    model_calls=0, backward_calls=0, optimizer_calls=0, cuda_initialized=torch.cuda.is_initialized(),
    limits=['This validates or corrects the existing target-only derivative carrier; it is not a reconstruction of an early complete historical minibatch.',
        'A target coefficient and input equality check does not prove the full85-iteration root cause. Native finite transport rounding remains reported.'])
assert report['cuda_initialized'] is False
path = OUT / 'historical-target-coefficient-and-native-loss-contract-v105.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))
