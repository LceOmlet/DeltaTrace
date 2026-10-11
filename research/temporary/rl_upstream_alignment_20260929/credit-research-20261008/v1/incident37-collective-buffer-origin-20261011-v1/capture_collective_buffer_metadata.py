"""Observe original root/layer0 FSDP buffers after an identified cross-parameter read.

Diagnostic only: call the installed owner unchanged, then retain Python tensor
metadata. No tensor values, copies, model calls, synchronization, or allocator
settings are added. The preceding failed GEMM row has been exactly identified
as a contiguous layer0 rank0 parameter payload, rather than an lm_head row.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import capture_native_incident37_GEMM_rows_20261011 as gemm


GEMM_SHA = '35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8'
COLLECTIVES_SHA = '3bf079219682d1a9439b81754af65e87f69157ec69f20f065e5565519f7093d9'
GROUP_SHA = '2798a87b2631091df738afaff6dfcead38ce0cc421585b4331ecaaefb4e784f5'
original_factory = gemm.make_observed_worker


def make_worker(native_factory):
    import ray
    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class CollectiveBufferMetadataObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from torch.distributed.fsdp._fully_shard import _fsdp_collectives as col
            from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

            for owner, expected in [(col, COLLECTIVES_SHA), (pg, GROUP_SHA)]:
                assert hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest() == expected
            output = Path(self._diagnostic_async_head_output)
            watched = {}
            for name, module in self.actor_module_fsdp.named_modules():
                if not hasattr(module, '_get_fsdp_state'):
                    continue
                group = module._get_fsdp_state()._fsdp_param_group
                if group is None or not group.fsdp_params:
                    continue
                names = [p._param_fqn for p in group.fsdp_params]
                if (any(n and n.endswith('lm_head.weight') for n in names)
                        or any(n and '.layers.0.' in n for n in names)):
                    watched[id(group.fsdp_params[0])] = (name, group, names)
            assert len(watched) == 2, list(watched.values())
            records = []
            owner_inputs = col._get_param_all_gather_inputs
            owner_gather = pg.foreach_all_gather
            owner_copy_out = pg.foreach_all_gather_copy_out

            def tensor_metadata(t):
                storage = t.untyped_storage()
                return dict(shape=list(t.shape), stride=list(t.stride()),
                            dtype=str(t.dtype), device=str(t.device),
                            data_ptr=t.data_ptr(), storage_ptr=storage.data_ptr(),
                            storage_bytes=storage.nbytes(),
                            storage_offset=t.storage_offset(), numel=t.numel())

            def identify(params):
                return watched.get(id(params[0])) if params else None

            def record(kind, watched_group, **values):
                name, group, names = watched_group
                records.append(dict(sequence=len(records), unix_ns=time.time_ns(),
                                    kind=kind, module=name,
                                    training_state=str(group._training_state),
                                    sharded_state=str(group._sharded_state), **values))

            @functools.wraps(owner_inputs)
            def observed_inputs(fsdp_params):
                result = owner_inputs(fsdp_params)
                target = identify(fsdp_params)
                if target is not None:
                    record('native_inputs_returned', target, parameters=[
                        dict(name=name, orig_size=list(p._orig_size),
                             sharded_source=tensor_metadata(p._sharded_param_data),
                             actual_inputs=[tensor_metadata(t) for t in ts])
                        for name, p, ts in zip(target[2], fsdp_params, result)])
                return result

            @functools.wraps(owner_gather)
            def observed_gather(fsdp_params, *args, **kwargs):
                result = owner_gather(fsdp_params, *args, **kwargs)
                target = identify(fsdp_params)
                if target is not None and result is not None:
                    record('native_gather_returned', target,
                           gather_output=tensor_metadata(result.all_gather_output),
                           input_numels=result.param_all_gather_input_numels,
                           input_split_sizes=result.all_gather_input_split_sizes,
                           input_dtypes=[[str(x) for x in row]
                                         for row in result.param_all_gather_input_dtypes])
                return result

            @functools.wraps(owner_copy_out)
            def observed_copy_out(all_gather_result, fsdp_params, *args, **kwargs):
                result = owner_copy_out(all_gather_result, fsdp_params, *args, **kwargs)
                target = identify(fsdp_params)
                if target is not None:
                    record('native_copy_out_returned', target,
                           gather_output=tensor_metadata(all_gather_result.all_gather_output),
                           parameters=[dict(name=name,
                               outputs=[tensor_metadata(t) for t in p.all_gather_outputs])
                               for name, p in zip(target[2], fsdp_params)])
                return result

            col._get_param_all_gather_inputs = observed_inputs
            pg.foreach_all_gather = observed_gather
            pg.foreach_all_gather_copy_out = observed_copy_out
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                col._get_param_all_gather_inputs = owner_inputs
                pg.foreach_all_gather = owner_gather
                pg.foreach_all_gather_copy_out = owner_copy_out
                report = dict(rank=self.rank, unix=time.time(), records=records,
                              source=str(Path(__file__)),
                              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              native_update_error=repr(failure) if failure is not None else None,
                              observed_groups=[dict(module=x[0], parameters=x[2])
                                               for x in watched.values()],
                              added_tensor_value_reads=0, added_tensor_copies=0,
                              added_model_calls=0, added_synchronizations=0,
                              owner_functions_restored=True, production_changes=0)
                (output / f'rank{self.rank}-collective-buffer-metadata.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(CollectiveBufferMetadataObserved)


def main():
    assert hashlib.sha256(Path(gemm.__file__).read_bytes()).hexdigest() == GEMM_SHA
    gemm.make_observed_worker = make_worker
    try:
        gemm.main()
    finally:
        gemm.make_observed_worker = original_factory


if __name__ == '__main__':
    main()
