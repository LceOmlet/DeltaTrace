"""Locate the first bad head row at actual native FSDP copy boundaries.

Compose the existing failing replay and its two-group lifecycle observer.
Only the first three root unshards are sampled: first forward, first backward,
second forward. No new model/operator implementation, training or acceptance
tolerance. Read one original packed row and its head output asynchronously.
Native record_stream protects the added packed-buffer read; this intentionally
perturbs that buffer's lifetime and is diagnostic only. A normal replay is not
evidence of a fix. All CPU value reads and event waits occur after update exits.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import capture_native_two_group_lifecycle as lifecycle

gemm = lifecycle.gemm
LIFECYCLE_SHA = 'f5ced5691c7c593637d2d34b2fb0f863fe2c70658f238e0ab70fcac511869358'
original_factory = lifecycle.make_worker


def make_worker(native_factory):
    import ray

    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class RootCopyBoundaryObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch

            roots = []
            for name, module in self.actor_module_fsdp.named_modules():
                if not hasattr(module, '_get_fsdp_state'):
                    continue
                group = module._get_fsdp_state()._fsdp_param_group
                if group is None:
                    continue
                for index, param in enumerate(group.fsdp_params):
                    if param._param_fqn and param._param_fqn.endswith('lm_head.weight'):
                        roots.append((name, group, index, param))
            assert len(roots) == 1
            label, group, head_index, head_param = roots[0]
            vocab_id = 159048  # Existing incident's exact competing row identity.
            width = head_param._orig_size[-1]
            buffers = torch.empty((3, 2, width), dtype=torch.bfloat16,
                                  device='cpu', pin_memory=True)
            records, pending, errors = [], [], []
            existed = 'wait_for_unshard' in vars(group)
            previous_local = vars(group).get('wait_for_unshard')
            owner_wait = group.wait_for_unshard

            @functools.wraps(owner_wait)
            def observed_wait(_group, *args, **kwargs):
                gathered = _group._all_gather_result
                result = owner_wait(*args, **kwargs)
                if gathered is None or len(records) >= 3:
                    return result
                # Do not read a pending collective: the original owner wait and
                # copy-out above already put their completion on this stream.
                position = len(records)
                record = dict(index=position, unix_ns=time.time_ns(), module=label,
                              training_state=str(_group._training_state))
                records.append(record)
                try:
                    split_sizes = gathered.all_gather_input_split_sizes
                    numels = gathered.param_all_gather_input_numels[head_index]
                    assert len(numels) == 1
                    shard_numel = numels[0]
                    head_offset = vocab_id * width
                    source_rank, local_offset = divmod(head_offset, shard_numel)
                    packed_offset = (source_rank * sum(split_sizes)
                                     + sum(split_sizes[:head_index]) + local_offset)
                    packed = gathered.all_gather_output
                    head = head_param.all_gather_outputs[0]
                    assert packed.dtype == head.dtype == buffers.dtype
                    stream = torch.cuda.current_stream()
                    record.update(vocab_id=vocab_id, width=width,
                        split_sizes=split_sizes, shard_numel=shard_numel,
                        packed_offset=packed_offset, head_offset=head_offset,
                        packed_pointer=packed.data_ptr(), head_pointer=head.data_ptr(),
                        stream=stream.cuda_stream)
                    # Protect only this diagnostic's cross-stream packed read.
                    # Head is allocated and consumed on the current stream.
                    packed.record_stream(stream)
                    buffers[position, 0].copy_(
                        packed.narrow(0, packed_offset, width), non_blocking=True)
                    buffers[position, 1].copy_(
                        head.narrow(0, head_offset, width), non_blocking=True)
                    event = stream.record_event()
                    pending.append((position, event))
                    record['copies_enqueued'] = True
                except Exception as error:
                    record['observation_error'] = repr(error)
                    errors.append(repr(error))
                return result

            group.wait_for_unshard = types.MethodType(observed_wait, group)
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                if existed:
                    group.wait_for_unshard = previous_local
                else:
                    delattr(group, 'wait_for_unshard')
                output = Path(self._diagnostic_async_head_output)
                ready = []
                for index, event in pending:
                    try:
                        event.synchronize()  # After native update exits only.
                        ready.append(index)
                        for col, key in [(0, 'packed'), (1, 'head')]:
                            value = buffers[index, col]
                            records[index][key + '_row_SHA256'] = hashlib.sha256(
                                bytes(value.view(torch.uint8).numpy())).hexdigest()
                        records[index]['packed_head_exact'] = torch.equal(
                            buffers[index, 0], buffers[index, 1])
                    except Exception as error:
                        errors.append(repr(error))
                artifact = output / f'rank{self.rank}-root-copy-boundary.pt'
                torch.save(dict(records=records, ready_indices=ready,
                                rows=buffers[ready].clone()), artifact)
                report = dict(rank=self.rank, unix=time.time(), records=records,
                    source=__file__, source_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()).hexdigest(),
                    native_update_error=repr(failure) if failure is not None else None,
                    errors=errors, ready_indices=ready,
                    artifact=dict(path=str(artifact), bytes=artifact.stat().st_size,
                        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                    added_model_calls=0, added_GPU_numeric_operations=0,
                    added_D2H_copies=len(pending) * 2,
                    added_packed_record_stream=len(pending),
                    pre_loss_explicit_synchronizations=0,
                    post_update_event_waits=len(pending),
                    observation_lifetime_perturbation=True,
                    production_changes=0, root_instance_method_restored=True)
                (output / f'rank{self.rank}-root-copy-boundary.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(RootCopyBoundaryObserved)


def main():
    assert hashlib.sha256(Path(lifecycle.__file__).read_bytes()).hexdigest() == LIFECYCLE_SHA
    lifecycle.make_worker = make_worker
    try:
        lifecycle.main()
    finally:
        lifecycle.make_worker = original_factory


if __name__ == '__main__':
    main()
