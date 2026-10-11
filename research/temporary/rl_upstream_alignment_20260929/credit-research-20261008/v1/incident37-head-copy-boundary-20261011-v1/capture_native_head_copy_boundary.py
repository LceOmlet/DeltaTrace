"""Read the native head row after three root unshards, without packed reads.

The prior packed/head diagnostic was finite, but added record_stream to protect
its packed read. This observer removes that added lifetime modification and
does not retain the communication result across the original wait. It reads
only the head row on the same stream that allocated/copied it. Existing saved
incident, two-group metadata and GEMM observation are composed unchanged.
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

    class HeadCopyBoundaryObserved(native_class):
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
                for param in group.fsdp_params:
                    if param._param_fqn and param._param_fqn.endswith('lm_head.weight'):
                        roots.append((name, group, param))
            assert len(roots) == 1
            label, group, head_param = roots[0]
            vocab_id = 159048
            width = head_param._orig_size[-1]
            buffers = torch.empty((3, width), dtype=torch.bfloat16,
                                  device='cpu', pin_memory=True)
            records, pending, errors = [], [], []
            existed = 'wait_for_unshard' in vars(group)
            previous_local = vars(group).get('wait_for_unshard')
            owner_wait = group.wait_for_unshard

            @functools.wraps(owner_wait)
            def observed_wait(_group, *args, **kwargs):
                had_gather = _group._all_gather_result is not None
                result = owner_wait(*args, **kwargs)
                if not had_gather or len(records) >= 3:
                    return result
                position = len(records)
                record = dict(index=position, unix_ns=time.time_ns(), module=label,
                              training_state=str(_group._training_state))
                records.append(record)
                try:
                    head = head_param.all_gather_outputs[0]
                    stream = torch.cuda.current_stream()
                    record.update(vocab_id=vocab_id, width=width,
                        head_offset=vocab_id * width, head_pointer=head.data_ptr(),
                        stream=stream.cuda_stream)
                    buffers[position].copy_(head.narrow(0, vocab_id * width, width),
                                            non_blocking=True)
                    pending.append((position, stream.record_event()))
                    record['copy_enqueued'] = True
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
                        value = buffers[index]
                        records[index]['head_row_SHA256'] = hashlib.sha256(
                            bytes(value.view(torch.uint8).numpy())).hexdigest()
                    except Exception as error:
                        errors.append(repr(error))
                artifact = output / f'rank{self.rank}-head-copy-boundary.pt'
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
                    added_D2H_copies=len(pending), added_record_stream=0,
                    retained_packed_communication_buffers=0,
                    pre_loss_explicit_synchronizations=0,
                    post_update_event_waits=len(pending),
                    timing_perturbation_from_observation=True,
                    production_changes=0, root_instance_method_restored=True)
                (output / f'rank{self.rank}-head-copy-boundary.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(HeadCopyBoundaryObserved)


def main():
    assert hashlib.sha256(Path(lifecycle.__file__).read_bytes()).hexdigest() == LIFECYCLE_SHA
    lifecycle.make_worker = make_worker
    try:
        lifecycle.main()
    finally:
        lifecycle.make_worker = original_factory


if __name__ == '__main__':
    main()
