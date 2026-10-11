"""Observe row159048 along the existing FSDP transport, without replacing it.

Compose the original GEMM and post-exit shard observers. Capture only the
first two lm_head unshards in native update_actor. Read the owner's actual
flat CPU source, returned H2D input, and unpacked all_gather_outputs. Queue
8 KiB D2H copies on their existing streams; wait only after update exits.
This is a saved-incident diagnostic, not a production implementation.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import capture_native_incident37_postfailure_shard_20261011 as post


POST_SHA = 'a5686e51dedbba9fab2039378b6acb1a42a91f3a7292ab4d5c38f530a73dc889'
VOCAB_ID = 159048


def make_observed_worker(native_factory):
    import ray
    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class NativeTransportObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from torch.distributed.fsdp._fully_shard import _fsdp_collectives as coll
            from torch.distributed.fsdp._fully_shard import _fsdp_param_group as groups

            names = {id(m): n for n, m in self.actor_module_fsdp.named_modules()}
            native_get = coll._get_param_all_gather_inputs
            native_copy = groups.foreach_all_gather_copy_out
            captures, errors, counters = [], [], {}
            source_events = {}

            def selected(params):
                return [(i, p, names.get(id(p._module_info.module), ''))
                        for i, p in enumerate(params)
                        if names.get(id(p._module_info.module), '').endswith('lm_head')
                        and p._module_info.param_name == 'weight']

            def queue(value, phase, index, p, name):
                record = dict(phase=phase, index=index, name=name,
                    dtype=str(value.dtype), device=str(value.device),
                    pointer=value.data_ptr(), shape=list(value.shape),
                    shard_rank=p.mesh_info.shard_mesh_rank,
                    stream=torch.cuda.current_stream().cuda_stream
                           if value.device.type != 'cpu' else None)
                if value.device.type == 'cpu':
                    record['buffer'] = value.detach().clone()
                else:
                    buffer = torch.empty(value.shape, dtype=value.dtype,
                                         device='cpu', pin_memory=True)
                    buffer.copy_(value.detach(), non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream())
                    record.update(buffer=buffer, event=event)
                captures.append(record)

            @functools.wraps(native_get)
            def observed_get(params):
                active = []
                for i, p, name in selected(params):
                    index = counters.get(id(p), 0)
                    counters[id(p)] = index + 1
                    source_events[id(p)] = index
                    if index >= 2:
                        continue
                    active.append((i, p, name, index))
                    try:
                        flat = p._sharded_param_data
                        width = p.padded_sharded_param_size[1]
                        local = VOCAB_ID - p.mesh_info.shard_mesh_rank * p.padded_sharded_param_size[0]
                        if 0 <= local < p.padded_sharded_param_size[0]:
                            queue(flat[local * width:(local + 1) * width],
                                  'actual_flat_source_before_H2D', index, p, name)
                    except Exception as error:
                        errors.append(dict(phase='flat_source', error=repr(error)))
                result = native_get(params)
                for i, p, name, index in active:
                    try:
                        width = p.padded_sharded_param_size[1]
                        local = VOCAB_ID - p.mesh_info.shard_mesh_rank * p.padded_sharded_param_size[0]
                        if 0 <= local < p.padded_sharded_param_size[0]:
                            if len(result[i]) != 1:
                                errors.append(dict(phase='native_H2D_input', error='Multiple native inputs; not reconstructed'))
                            else:
                                queue(result[i][0][local * width:(local + 1) * width],
                                      'native_H2D_input', index, p, name)
                    except Exception as error:
                        errors.append(dict(phase='native_H2D_input', error=repr(error)))
                return result

            @functools.wraps(native_copy)
            def observed_copy(result, params, group):
                output = native_copy(result, params, group)
                for _, p, name in selected(params):
                    index = source_events.get(id(p), 2)
                    if index >= 2:
                        continue
                    try:
                        if len(p.all_gather_outputs) != 1:
                            errors.append(dict(phase='native_unpacked_output', error='Multiple native outputs; not reconstructed'))
                        else:
                            width = p.padded_sharded_param_size[1]
                            value = p.all_gather_outputs[0]
                            queue(value[VOCAB_ID * width:(VOCAB_ID + 1) * width],
                                  'native_unpacked_all_gather_output', index, p, name)
                    except Exception as error:
                        errors.append(dict(phase='native_unpacked_output', error=repr(error)))
                return output

            coll._get_param_all_gather_inputs = observed_get
            groups.foreach_all_gather_copy_out = observed_copy
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                coll._get_param_all_gather_inputs = native_get
                groups.foreach_all_gather_copy_out = native_copy
                for record in captures:
                    event = record.pop('event', None)
                    if event is not None:
                        event.synchronize()
                output = Path(self._diagnostic_async_head_output)
                path = output / f'rank{self.rank}-transport.pt'
                torch.save(captures, path)
                report = dict(rank=self.rank, unix=time.time(), source=__file__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_update_error=repr(failure) if failure is not None else None,
                    errors=errors, observations=[{k: v for k, v in r.items() if k != 'buffer'} for r in captures],
                    artifact=dict(path=str(path), bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    extra_collectives=0, extra_model_calls=0, extra_GEMM_calls=0,
                    pre_exit_synchronizations=0, original_owners_restored=True,
                    production_changes=0)
                (output / f'rank{self.rank}-transport.json').write_text(json.dumps(report, indent=2) + '\n')

    return ray.remote(NativeTransportObserved)


original_factory = post.make_observed_worker


def main():
    assert hashlib.sha256(Path(post.__file__).read_bytes()).hexdigest() == POST_SHA
    post.make_observed_worker = make_observed_worker
    try:
        post.main()
    finally:
        post.make_observed_worker = original_factory


if __name__ == '__main__':
    main()
