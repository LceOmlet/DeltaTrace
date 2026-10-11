"""Compose the captured native stages with CPU-source/owner-state observations.

No new stream operations, event queries, device synchronization, parameter
transition or arithmetic. The already failing stage observer is unchanged.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import capture_root_native_stages as stages

STAGES_SHA = '5a4d777b1cfd34df9e559a2e8d4e040303281a7539bc3f14fa1bb2789b902dde'
original_factory = stages.make_worker


def make_worker(native_factory):
    import ray
    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class NativeDependenciesObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from torch.distributed.fsdp._fully_shard import _fsdp_collectives as c

            roots, layers = [], []
            for name, module in self.actor_module_fsdp.named_modules():
                if not hasattr(module, '_get_fsdp_state'):
                    continue
                state = module._get_fsdp_state()
                group = state._fsdp_param_group
                if group is None:
                    continue
                for p in group.fsdp_params:
                    if p._param_fqn and p._param_fqn.endswith('lm_head.weight'):
                        roots.append((state, group, p))
                if any(p._param_fqn and '.layers.0.' in p._param_fqn for p in group.fsdp_params):
                    layers.append(group)
            assert len(roots) == len(layers) == 1
            state, root, head = roots[0]
            layer = layers[0]
            records, rows, installed = [], {}, []

            def record(kind):
                pending = layer._all_gather_result
                context = state._state_ctx
                records.append(dict(unix_ns=time.time_ns(), kind=kind,
                    current_stream=torch.cuda.current_stream().cuda_stream,
                    copy_in_stream=root.comm_ctx.all_gather_copy_in_stream.cuda_stream,
                    gather_stream=root.comm_ctx.all_gather_stream.cuda_stream,
                    iter_forward_root_is_None=context.iter_forward_root is None,
                    iter_forward_root_is_this_state=context.iter_forward_root is state,
                    post_optim_event_id=id(context.post_optim_event) if context.post_optim_event is not None else None,
                    root_training_state=str(root._training_state),
                    layer0_training_state=str(layer._training_state),
                    pending_layer0_pointer=pending.all_gather_output.data_ptr() if pending else None,
                    pending_layer0_event_id=id(pending.all_gather_event) if pending and pending.all_gather_event is not None else None,
                    pending_layer0_work=pending.all_gather_work is not None if pending else None))

            def install(obj, method, before, after):
                owner = getattr(obj, method)
                existed = method in vars(obj)
                previous = vars(obj).get(method)

                @functools.wraps(owner)
                def observed(_object, *args, **kwargs):
                    record(before)
                    result = owner(*args, **kwargs)
                    record(after)
                    return result

                setattr(obj, method, types.MethodType(observed, obj))
                installed.append((obj, method, existed, previous))

            owner_inputs = c._get_param_all_gather_inputs
            @functools.wraps(owner_inputs)
            def inputs(params):
                if params is root.fsdp_params and str(root._training_state) == 'TrainingState.FORWARD':
                    local = 159048 - head.mesh_info.shard_mesh_rank * head.padded_sharded_param_size[0]
                    if 0 <= local < head.padded_sharded_param_size[0] and len(rows) < 2:
                        value = head._sharded_param_data
                        assert value.device.type == 'cpu'
                        width = head._orig_size[-1]
                        rows[str(len(rows))] = value.reshape(-1)[local*width:(local+1)*width].detach().clone()
                        record('CPU_source_before_native_H2D')
                return owner_inputs(params)

            install(state, '_root_pre_forward', 'before_root_pre_forward', 'after_root_pre_forward')
            install(layer, 'finalize_backward', 'before_layer0_finalize', 'after_layer0_finalize')
            c._get_param_all_gather_inputs = inputs
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                c._get_param_all_gather_inputs = owner_inputs
                for obj, method, existed, previous in reversed(installed):
                    if existed:
                        setattr(obj, method, previous)
                    else:
                        delattr(obj, method)
                output = Path(self._diagnostic_async_head_output)
                artifact = output / f'rank{self.rank}-native-dependencies-CPU.pt'
                torch.save(rows, artifact)
                report = dict(rank=self.rank, unix=time.time(), source=__file__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_update_error=repr(failure) if failure is not None else None,
                    records=records, CPU_row_SHA256={k:hashlib.sha256(bytes(v.view(torch.uint8).numpy())).hexdigest() for k,v in rows.items()},
                    artifact=dict(path=str(artifact), bytes=artifact.stat().st_size,
                        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                    added_GPU_operations=0, added_synchronizations=0,
                    owner_stream_functions_unchanged=True, production_changes=0)
                (output / f'rank{self.rank}-native-dependencies.json').write_text(json.dumps(report, indent=2) + '\n')

    return ray.remote(NativeDependenciesObserved)


def main():
    assert hashlib.sha256(Path(stages.__file__).read_bytes()).hexdigest() == STAGES_SHA
    stages.make_worker = make_worker
    try:
        stages.main()
    finally:
        stages.make_worker = original_factory


if __name__ == '__main__':
    main()
