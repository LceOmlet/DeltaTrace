"""Record native root/layer0 collective pointers in the saved real incident.

The preceding full metadata observer did not reproduce the failure. This
observer keeps only scalar buffer addresses and byte lengths, rather than
walking every input/output tensor. Native functions and the existing GEMM
capture remain unchanged. No values, copies, waits or model calls are added.
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

    class CollectivePointersObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            from torch.distributed.fsdp._fully_shard import _fsdp_collectives as col
            from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

            for owner, expected in [(col, COLLECTIVES_SHA), (pg, GROUP_SHA)]:
                assert hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest() == expected
            watched = {}
            for name, module in self.actor_module_fsdp.named_modules():
                if not hasattr(module, '_get_fsdp_state'):
                    continue
                group = module._get_fsdp_state()._fsdp_param_group
                if group is None or not group.fsdp_params:
                    continue
                names = [p._param_fqn for p in group.fsdp_params]
                heads = [i for i, n in enumerate(names) if n and n.endswith('lm_head.weight')]
                if heads or any(n and '.layers.0.' in n for n in names):
                    watched[id(group.fsdp_params[0])] = (name, heads[0] if heads else None)
            assert len(watched) == 2
            records = []
            owner_gather = pg.foreach_all_gather
            owner_copy_out = pg.foreach_all_gather_copy_out

            @functools.wraps(owner_gather)
            def observed_gather(params, *args, **kwargs):
                result = owner_gather(params, *args, **kwargs)
                target = watched.get(id(params[0])) if params else None
                if target is not None and result is not None:
                    t = result.all_gather_output
                    records.append((time.time_ns(), 'gather', target[0],
                                    t.data_ptr(), t.untyped_storage().nbytes()))
                return result

            @functools.wraps(owner_copy_out)
            def observed_copy_out(result, params, *args, **kwargs):
                value = owner_copy_out(result, params, *args, **kwargs)
                target = watched.get(id(params[0])) if params else None
                if target is not None and target[1] is not None:
                    t = params[target[1]].all_gather_outputs[0]
                    records.append((time.time_ns(), 'head_copy_out', target[0],
                                    t.data_ptr(), t.untyped_storage().nbytes()))
                return value

            pg.foreach_all_gather = observed_gather
            pg.foreach_all_gather_copy_out = observed_copy_out
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                pg.foreach_all_gather = owner_gather
                pg.foreach_all_gather_copy_out = owner_copy_out
                report = dict(rank=self.rank, unix=time.time(),
                              record_fields=['unix_ns', 'kind', 'module', 'data_ptr', 'storage_bytes'],
                              records=records,
                              source=str(Path(__file__)),
                              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              native_update_error=repr(failure) if failure is not None else None,
                              added_tensor_value_reads=0, added_tensor_copies=0,
                              added_model_calls=0, added_synchronizations=0,
                              owner_functions_restored=True, production_changes=0)
                output = Path(self._diagnostic_async_head_output)
                (output / f'rank{self.rank}-collective-pointers.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(CollectivePointersObserved)


def main():
    assert hashlib.sha256(Path(gemm.__file__).read_bytes()).hexdigest() == GEMM_SHA
    gemm.make_observed_worker = make_worker
    try:
        gemm.main()
    finally:
        gemm.make_observed_worker = original_factory


if __name__ == '__main__':
    main()
