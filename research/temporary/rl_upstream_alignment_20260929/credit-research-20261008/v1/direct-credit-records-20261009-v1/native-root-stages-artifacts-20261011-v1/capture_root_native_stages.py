"""Read the incident's head row at existing FSDP boundaries, without replacing them.

Only two actual FORWARD root gathers are observed. Each added D2H read is
enqueued on the owner's current stream before its existing completion event;
no added record_stream, pre-update synchronize, GPU arithmetic or retained GPU
tensor. Timing is still perturbed: a finite replay is not evidence of repair.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import capture_native_two_group_lifecycle as lifecycle

original_factory = lifecycle.make_worker
LIFECYCLE_SHA = 'f5ced5691c7c593637d2d34b2fb0f863fe2c70658f238e0ab70fcac511869358'
SELECTION = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-incident37-competing-weight-capture-20261011-v1/selection.json')
SELECTION_SHA = '9a82a6912a18e44cf11cca67bcc217ffbcdf6d16dbc8adc71bd487f461c5cd1c'


def make_worker(native_factory):
    import ray
    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class RootNativeStagesObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from torch.distributed.fsdp._fully_shard import _fsdp_collectives as c
            from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

            roots = []
            for name, module in self.actor_module_fsdp.named_modules():
                if not hasattr(module, '_get_fsdp_state'):
                    continue
                group = module._get_fsdp_state()._fsdp_param_group
                if group is not None:
                    for i, p in enumerate(group.fsdp_params):
                        if p._param_fqn and p._param_fqn.endswith('lm_head.weight'):
                            roots.append((name, group, i, p))
            assert len(roots) == 1
            name, root, head_index, head = roots[0]
            width, vocab = head._orig_size[-1], 159048
            host = torch.empty((2, 4, width), dtype=torch.bfloat16, device='cpu', pin_memory=True)
            records, copies, events, errors = [], [], [], []
            current = None
            forward_count = 0
            selected_outputs = {}
            owner_unshard = root.unshard
            existed = 'unshard' in vars(root)
            previous = vars(root).get('unshard')
            owner_inputs = c._get_param_all_gather_inputs
            owner_copy_in = torch.ops.fsdp.all_gather_copy_in
            owner_copy_out = pg.foreach_all_gather_copy_out

            def sample(slot, col, tensor, offset, label):
                stream = torch.cuda.current_stream()
                host[slot, col].copy_(tensor.reshape(-1).narrow(0, offset, width), non_blocking=True)
                copies.append((slot, col))
                events.append(stream.record_event())
                records.append(dict(slot=slot, stage=label, column=col,
                    unix_ns=time.time_ns(), pointer=tensor.data_ptr(), offset=offset,
                    storage_bytes=tensor.untyped_storage().nbytes(), stream=stream.cuda_stream))

            @functools.wraps(owner_inputs)
            def inputs(params):
                result = owner_inputs(params)
                if current is not None and params is root.fsdp_params:
                    local = vocab - head.mesh_info.shard_mesh_rank * head.padded_sharded_param_size[0]
                    if 0 <= local < head.padded_sharded_param_size[0]:
                        assert len(result[head_index]) == 1
                        sample(current, 0, result[head_index][0], local * width, 'after_native_H2D')
                return result

            @functools.wraps(owner_copy_in)
            def copy_in(*args, **kwargs):
                result = owner_copy_in(*args, **kwargs)
                if current is not None:
                    splits, numel, rank = args[1], args[2], args[4]
                    assert numel == sum(splits)
                    selected_outputs[result[1].data_ptr()] = current
                    local = vocab - head.mesh_info.shard_mesh_rank * head.padded_sharded_param_size[0]
                    if 0 <= local < head.padded_sharded_param_size[0]:
                        sample(current, 1, result[0], sum(splits[:head_index]) + local * width,
                               'after_native_copy_in')
                    records.append(dict(slot=current, stage='native_copy_in_metadata', rank=rank,
                        split_sizes=list(splits), input_pointer=result[0].data_ptr(),
                        output_pointer=result[1].data_ptr(), input_numel=numel))
                return result

            @functools.wraps(owner_copy_out)
            def copy_out(gathered, params, group):
                result = owner_copy_out(gathered, params, group)
                slot = (selected_outputs.pop(gathered.all_gather_output.data_ptr(), None)
                        if params is root.fsdp_params else None)
                if slot is not None and params is root.fsdp_params:
                    sizes = gathered.all_gather_input_split_sizes
                    shard_numel = gathered.param_all_gather_input_numels[head_index][0]
                    source_rank, local_offset = divmod(vocab * width, shard_numel)
                    offset = source_rank * sum(sizes) + sum(sizes[:head_index]) + local_offset
                    sample(slot, 2, gathered.all_gather_output, offset, 'after_native_collective')
                    sample(slot, 3, head.all_gather_outputs[0], vocab * width, 'after_native_copy_out')
                return result

            @functools.wraps(owner_unshard)
            def unshard(_group, *args, **kwargs):
                nonlocal current, forward_count
                selected = (_group._all_gather_result is None and not _group.is_unsharded
                            and str(_group._training_state) == 'TrainingState.FORWARD')
                slot = forward_count if selected else None
                if selected:
                    forward_count += 1
                current = slot if slot is not None and slot < 2 else None
                try:
                    return owner_unshard(*args, **kwargs)
                finally:
                    current = None

            root.unshard = types.MethodType(unshard, root)
            c._get_param_all_gather_inputs = inputs
            torch.ops.fsdp.all_gather_copy_in = copy_in
            pg.foreach_all_gather_copy_out = copy_out
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                c._get_param_all_gather_inputs = owner_inputs
                torch.ops.fsdp.all_gather_copy_in = owner_copy_in
                pg.foreach_all_gather_copy_out = owner_copy_out
                if existed:
                    root.unshard = previous
                else:
                    delattr(root, 'unshard')
                output = Path(self._diagnostic_async_head_output)
                ready = []
                for position, event in enumerate(events):
                    try:
                        event.synchronize()  # Original update has already exited.
                        ready.append(copies[position])
                    except Exception as error:
                        errors.append(repr(error))
                rows = {f'{slot}:{col}': host[slot, col].clone() for slot, col in ready}
                local = vocab - head.mesh_info.shard_mesh_rank * head.padded_sharded_param_size[0]
                source = head._sharded_param_data
                if source.device.type == 'cpu' and 0 <= local < head.padded_sharded_param_size[0]:
                    rows['terminal_CPU'] = source.reshape(-1)[local*width:(local+1)*width].detach().clone()
                hashes = {k: hashlib.sha256(bytes(v.view(torch.uint8).numpy())).hexdigest()
                          for k, v in rows.items()}
                artifact = output / f'rank{self.rank}-root-native-stages.pt'
                torch.save(rows, artifact)
                report = dict(rank=self.rank, unix=time.time(), module=name, vocab_id=vocab,
                    source=__file__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_update_error=repr(failure) if failure is not None else None,
                    records=records, ready=ready, row_SHA256=hashes, errors=errors,
                    artifact=dict(path=str(artifact), bytes=artifact.stat().st_size,
                        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                    added_GPU_arithmetic=0, added_model_calls=0, added_record_stream=0,
                    added_D2H_copies=len(copies), pre_update_synchronizations=0,
                    post_update_event_waits=len(events), production_changes=0,
                    original_functions_restored=True, timing_perturbed=True)
                (output / f'rank{self.rank}-root-native-stages.json').write_text(json.dumps(report, indent=2) + '\n')

    return ray.remote(RootNativeStagesObserved)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    assert hashlib.sha256(Path(lifecycle.__file__).read_bytes()).hexdigest() == LIFECYCLE_SHA
    selection = SELECTION.read_bytes()
    assert hashlib.sha256(selection).hexdigest() == SELECTION_SHA
    target = args.output.parent / 'selection.json'
    if target.exists():
        assert target.read_bytes() == selection
    else:
        target.write_bytes(selection)
    lifecycle.make_worker = make_worker
    try:
        lifecycle.main()
    finally:
        lifecycle.make_worker = original_factory


if __name__ == '__main__':
    main()
