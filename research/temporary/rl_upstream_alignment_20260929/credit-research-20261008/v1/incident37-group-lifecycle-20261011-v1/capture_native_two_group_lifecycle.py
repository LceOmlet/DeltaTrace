"""Observe two actual FSDP groups in the saved failing native incident.

The prior global collective wrappers ran on every layer and did not reproduce
the defect. This diagnostic wraps only the root and decoder0 group instances.
It records scalar pointers at native gather, copy-out, and release boundaries.
It adds no tensor operations, synchronization, or training implementation.
The existing failed-replay GEMM/gradient capture remains unchanged.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import capture_native_incident37_GEMM_rows_20261011 as gemm

GEMM_SHA = '35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8'
GROUP_SHA = '2798a87b2631091df738afaff6dfcead38ce0cc421585b4331ecaaefb4e784f5'
original_factory = gemm.make_observed_worker


def make_worker(native_factory):
    import ray
    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class TwoGroupLifecycleObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

            assert hashlib.sha256(Path(pg.__file__).read_bytes()).hexdigest() == GROUP_SHA
            watched, installed, records = [], [], []
            for name, module in self.actor_module_fsdp.named_modules():
                if not hasattr(module, '_get_fsdp_state'):
                    continue
                group = module._get_fsdp_state()._fsdp_param_group
                if group is None or not group.fsdp_params:
                    continue
                names = [p._param_fqn for p in group.fsdp_params]
                heads = [i for i, n in enumerate(names) if n and n.endswith('lm_head.weight')]
                if heads or any(n and '.layers.0.' in n for n in names):
                    watched.append((name, group, heads[0] if heads else None))
            assert len(watched) == 2

            def observe(label, group, head_index, kind):
                gather = group._all_gather_result
                stage = gather.all_gather_output if gather is not None else None
                head_outputs = (group.fsdp_params[head_index].all_gather_outputs
                                if head_index is not None else [])
                head = head_outputs[0] if head_outputs else None
                records.append((time.time_ns(), label, kind,
                    str(group._training_state), str(group._sharded_state),
                    stage.data_ptr() if stage is not None else None,
                    stage.untyped_storage().nbytes() if stage is not None else None,
                    head.data_ptr() if head is not None else None,
                    head.untyped_storage().nbytes() if head is not None else None))

            def install(label, group, head_index, method_name):
                existed = method_name in vars(group)
                previous_local = vars(group).get(method_name)
                owner_method = getattr(group, method_name)

                @functools.wraps(owner_method)
                def observed(_group, *args, **kwargs):
                    if method_name == 'reshard':
                        observe(label, _group, head_index, 'before_reshard')
                    result = owner_method(*args, **kwargs)
                    if method_name != 'reshard':
                        observe(label, _group, head_index, 'after_' + method_name)
                    return result

                setattr(group, method_name, types.MethodType(observed, group))
                installed.append((group, method_name, existed, previous_local))

            try:
                for label, group, head_index in watched:
                    for method_name in ('unshard', 'wait_for_unshard', 'reshard'):
                        install(label, group, head_index, method_name)
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                for group, name, existed, previous in reversed(installed):
                    if existed:
                        setattr(group, name, previous)
                    else:
                        delattr(group, name)
                output = Path(self._diagnostic_async_head_output)
                report = dict(rank=self.rank, unix=time.time(),
                    source=__file__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    record_fields=['unix_ns', 'module', 'boundary', 'training_state',
                        'sharded_state', 'gather_pointer', 'gather_storage_bytes',
                        'head_pointer', 'head_storage_bytes'],
                    watched_modules=[label for label, _, _ in watched],
                    records=records,
                    native_update_error=repr(failure) if failure is not None else None,
                    added_tensor_value_operations=0, added_synchronizations=0,
                    added_model_calls=0, production_changes=0,
                    instance_methods_restored=True)
                (output / f'rank{self.rank}-two-group-lifecycle.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(TwoGroupLifecycleObserved)


def main():
    assert hashlib.sha256(Path(gemm.__file__).read_bytes()).hexdigest() == GEMM_SHA
    gemm.make_observed_worker = make_worker
    try:
        gemm.main()
    finally:
        gemm.make_observed_worker = original_factory


if __name__ == '__main__':
    main()
