"""Compose native sharded state loading with the existing derivative observer.

Only a diagnostic parameter partition is loaded before the unchanged native
derivative calls. There is no model/loss/optimizer/backward implementation here.
"""
import hashlib
from pathlib import Path


def load_diagnostic_partition(worker, case):
    import hashlib
    import inspect
    import json
    import torch
    from pathlib import Path
    from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
    from verl.utils.fsdp_utils import get_fsdp_state_ctx
    from observe_native_gradient_chain_v4 import digest

    spec = case['diagnostic_parameter_partition']
    artifact = spec['rank_artifacts'][str(worker.rank)]
    p = Path(artifact['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == artifact['SHA256']
    before = digest(worker.actor)
    assert before == artifact['expected_base_trainable_SHA256']
    partial = torch.load(p, map_location='cpu', weights_only=False)
    assert list(partial) == artifact['selected_keys']
    parameters = dict(worker.actor.actor_module.named_parameters())
    assert set(partial) <= parameters.keys()
    for name, value in partial.items():
        assert parameters[name].requires_grad
        assert value.shape == parameters[name].shape and value.dtype == parameters[name].dtype
    scheduler = worker.actor_lr_scheduler.last_epoch
    optimizer = worker.actor.actor_optimizer
    steps = sorted({float(s['step']) for s in optimizer.state.values() if 'step' in s})
    model = worker.checkpoint_manager.model
    assert model is worker.actor.actor_module
    # Same public sharded-state context and Torch loader used by the pinned
    # checkpoint owner. strict=False is the owner-supported partial-dict mode.
    with get_fsdp_state_ctx(model, StateDictType.SHARDED_STATE_DICT,
                            ShardedStateDictConfig(offload_to_cpu=True),
                            ShardedOptimStateDictConfig(offload_to_cpu=True)):
        incompatible = model.load_state_dict(partial, strict=False)
    assert not incompatible.unexpected_keys
    assert not set(partial).intersection(incompatible.missing_keys)
    after = digest(worker.actor)
    assert after == artifact['expected_result_trainable_SHA256']
    assert worker.actor_lr_scheduler.last_epoch == scheduler
    assert sorted({float(s['step']) for s in optimizer.state.values() if 'step' in s}) == steps
    owner = Path(inspect.getsourcefile(get_fsdp_state_ctx))
    torch_owner = Path(inspect.getsourcefile(torch.nn.Module.load_state_dict))
    return dict(rank=worker.rank, case=case['name'], artifact_path=str(p),
        artifact_SHA256=artifact['SHA256'], base_trainable_SHA256=before,
        result_trainable_SHA256=after, selected_parameter_count=len(partial),
        expected_missing_nonpartition_keys=len(incompatible.missing_keys), unexpected_keys=[],
        scheduler_before=scheduler, scheduler_after=worker.actor_lr_scheduler.last_epoch,
        optimizer_steps_before=steps, optimizer_steps_after=steps,
        state_context_owner=dict(path=str(owner), SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
        torch_state_loader_owner=dict(path=str(torch_owner), SHA256=hashlib.sha256(torch_owner.read_bytes()).hexdigest()),
        model_forward_backward_optimizer_scheduler_calls=0)


source = Path(__file__).with_name('observe_native_gradient_chain_v4.py')
expected = '3cbefaf36f2a33712094cd017f0cdf026d1a94fee44d2d1c973eb74ef3251117'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
code = source.read_text()
old = '            bindings=group.set_case(iteration)'
new = '''            partition_bindings = ray.get([w.execute_with_func_generator.remote(
                func=load_diagnostic_partition, case=case) for w in group.workers])
            state.setdefault('diagnostic_parameter_partition_bindings', []).extend(partition_bindings)
            bindings=group.set_case(iteration)'''
assert code.count(old) == 1
code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
