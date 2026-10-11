"""Record the native CPU-offload gather inputs in the saved incident replay.

Compose the already failing two-group lifecycle observer. Only root/layer0
parameter-input addresses are added, immediately after the installed owner
returns them and before its original packing call. No tensor values, copies,
synchronization, retained tensor buffers, or numerical changes are introduced.
Python observation still changes timing; a finite replay is not a repair.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import capture_native_two_group_lifecycle as lifecycle

gemm = lifecycle.gemm
LIFECYCLE_SHA = 'f5ced5691c7c593637d2d34b2fb0f863fe2c70658f238e0ab70fcac511869358'
COLLECTIVES_SHA = '3bf079219682d1a9439b81754af65e87f69157ec69f20f065e5565519f7093d9'
original_factory = lifecycle.make_worker


def make_worker(native_factory):
    import ray

    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class ParameterInputPointersObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from torch.distributed.fsdp._fully_shard import _fsdp_collectives as col

            assert hashlib.sha256(Path(col.__file__).read_bytes()).hexdigest() == COLLECTIVES_SHA
            watched, head_param = {}, None
            for label, module in self.actor_module_fsdp.named_modules():
                if not hasattr(module, '_get_fsdp_state'):
                    continue
                group = module._get_fsdp_state()._fsdp_param_group
                if group is None or not group.fsdp_params:
                    continue
                names = [p._param_fqn for p in group.fsdp_params]
                heads = [p for p in group.fsdp_params
                         if p._param_fqn and p._param_fqn.endswith('lm_head.weight')]
                if heads or any(n and '.layers.0.' in n for n in names):
                    watched[id(group.fsdp_params[0])] = (label, group, names)
                if heads:
                    assert head_param is None and len(heads) == 1
                    head_param = heads[0]
            assert len(watched) == 2 and head_param is not None
            records = []
            owner_inputs = col._get_param_all_gather_inputs

            @functools.wraps(owner_inputs)
            def observed_inputs(fsdp_params):
                result = owner_inputs(fsdp_params)
                target = watched.get(id(fsdp_params[0])) if fsdp_params else None
                if target is not None:
                    label, group, names = target
                    head_outputs = head_param.all_gather_outputs
                    head = head_outputs[0] if head_outputs else None
                    head_storage = head.untyped_storage() if head is not None else None
                    records.append(dict(
                        unix_ns=time.time_ns(), module=label,
                        training_state=str(group._training_state),
                        stream=torch.cuda.current_stream().cuda_stream,
                        head_pointer=head.data_ptr() if head is not None else None,
                        head_storage_bytes=head_storage.nbytes() if head_storage is not None else None,
                        # Scalar metadata only: do not retain any input tensors.
                        inputs=[dict(parameter_index=i, input_index=j,
                                     data_ptr=t.data_ptr(),
                                     storage_ptr=t.untyped_storage().data_ptr(),
                                     storage_bytes=t.untyped_storage().nbytes(),
                                     numel=t.numel(), dtype=str(t.dtype))
                                for i, ts in enumerate(result)
                                for j, t in enumerate(ts)]))
                return result

            col._get_param_all_gather_inputs = observed_inputs
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                col._get_param_all_gather_inputs = owner_inputs
                output = Path(self._diagnostic_async_head_output)
                report = dict(rank=self.rank, unix=time.time(), records=records,
                    groups=[dict(module=x[0], parameter_names=x[2]) for x in watched.values()],
                    source=__file__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_update_error=repr(failure) if failure is not None else None,
                    added_tensor_value_reads=0, added_tensor_copies=0,
                    added_model_calls=0, added_synchronizations=0,
                    retained_parameter_input_tensors=0, added_record_stream=0,
                    timing_perturbation_from_observation=True,
                    owner_function_restored=True, production_changes=0)
                (output / f'rank{self.rank}-parameter-input-pointers.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(ParameterInputPointersObserved)


def main():
    assert hashlib.sha256(Path(lifecycle.__file__).read_bytes()).hexdigest() == LIFECYCLE_SHA
    lifecycle.make_worker = make_worker
    try:
        lifecycle.main()
    finally:
        lifecycle.make_worker = original_factory


if __name__ == '__main__':
    main()
