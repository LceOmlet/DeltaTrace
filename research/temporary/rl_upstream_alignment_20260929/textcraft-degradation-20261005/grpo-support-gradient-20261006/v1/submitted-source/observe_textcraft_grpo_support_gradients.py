"""Select saved GRPO support, then call the unchanged native loss observer.

Only diagnostic_grpo_advantages is rebound. DT A and every original mask,
old/ref log probability and token ID remain the saved native tensors. This
module never implements a loss, gradient aggregate or optimizer operation.
"""
import importlib
import json
from pathlib import Path


SUPPORTS = ('q_nonzero', 'q_zero')
LABELS = ('dt_pg', 'grpo_pg')


def support_snapshot(data, support):
    import torch
    from observe_textcraft_native_batches import _snapshot

    if support not in SUPPORTS:
        raise ValueError(support)
    selected = data.batch['dt_q_estimates'] != 0 if support == 'q_nonzero' else data.batch['dt_q_estimates'] == 0
    snapshot = _snapshot(data)
    snapshot.batch['diagnostic_grpo_advantages'] = torch.where(
        selected, data.batch['diagnostic_grpo_advantages'],
        torch.zeros_like(data.batch['diagnostic_grpo_advantages']))
    snapshot.meta_info['grpo_gradient_support'] = support
    return snapshot


def observe_saved_grpo_support(worker, data, *, original_update_actor, output_directory):
    """Return the original worker output; all diagnostic overrides are scoped."""
    import torch

    owner = importlib.import_module('observe_native_optimizer_minibatch')
    if owner.observe_native_optimizer_minibatch.__globals__ is not vars(owner):
        raise RuntimeError('The imported original observer must use its actual module LABELS.')
    support = data.meta_info['grpo_gradient_support']
    if support not in SUPPORTS:
        raise ValueError(support)
    selected = data.batch['dt_q_estimates'] != 0 if support == 'q_nonzero' else data.batch['dt_q_estimates'] == 0
    if torch.count_nonzero(data.batch['diagnostic_grpo_advantages'][~selected]).item():
        raise ValueError('GRPO advantages outside the selected saved Q support were not zeroed.')
    path = Path(output_directory) / f'rank{worker.rank}-{support}-gradients.json'
    scheduler_calls = []

    def observed_update_policy(*, data):
        return owner.observe_native_optimizer_minibatch(
            worker.actor, data, output_path=path, rank=worker.rank)

    def no_op_scheduler_step(*args, **kwargs):
        scheduler_calls.append(True)
        return None

    with owner._temporary_attribute(owner, 'LABELS', LABELS):
        with owner._temporary_attribute(worker.actor, 'update_policy', observed_update_policy):
            with owner._temporary_attribute(worker.actor_lr_scheduler, 'step', no_op_scheduler_step):
                output = original_update_actor(data)
    result = json.loads(path.read_bytes())
    result['original_observer_scope'] = result['scope']
    result['scope'] = ('One original local32/B4 complete minibatch; two native backward passes '
                       'with complete DT A and support-selected saved GRPO A. No optimizer/scheduler update.')
    result['grpo_advantage_provenance'] = (
        'Saved official global-UID GRPO advantages; only positions outside the specified saved Q support zeroed. '
        'Original full loss_mask and B4/8 denominator retained.')
    result['grpo_support'] = support
    result['actual_backward_passes'] = len(result['passes'])
    result['scheduler_step_no_op_calls'] = len(scheduler_calls)
    result['scheduler_step_executed'] = False
    result['diagnostic_adapter'] = owner._source_identity(observe_saved_grpo_support)
    result['support_count'] = int(selected.sum().item())
    result['rng_scope'] = ('Original observer restores Python/NumPy/Torch RNG and replays each pass from '
                           'its saved initial RNG. Repeated DT is descriptive, without an equality threshold.')
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return output
