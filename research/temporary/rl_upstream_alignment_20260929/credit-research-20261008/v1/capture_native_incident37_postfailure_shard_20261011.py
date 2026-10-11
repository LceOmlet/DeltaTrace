"""Read the native FSDP source shard only after an observed update exits.

Compose the already failing GEMM observer unchanged. Add no reads, copies,
hooks or synchronizations before its native update has returned or raised.
Do not read the released full weight; preserve the original exception.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import capture_native_incident37_GEMM_rows_20261011 as gemm

GEMM_SHA = '35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8'
VOCAB_ID = 159048


def make_observed_worker(native_factory):
    import ray
    actor = original_gemm_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class NativePostfailureShardObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                import torch
                rows, errors = [], []
                try:
                    names = {id(m): n for n, m in self.actor_module_fsdp.named_modules()}
                    for module in self.actor_module_fsdp.modules():
                        get_state = getattr(module, '_get_fsdp_state', None)
                        if get_state is None:
                            continue
                        group = get_state()._fsdp_param_group
                        if group is None:
                            continue
                        for p in group.fsdp_params:
                            info = p._module_info
                            name = names.get(id(info.module), '')
                            if not name.endswith('lm_head') or info.param_name != 'weight':
                                continue
                            value = p._sharded_local_tensor
                            rank = p.mesh_info.shard_mesh_rank
                            lo = rank * p.padded_sharded_param_size[0]
                            full = getattr(p, '_unsharded_param', None)
                            row = dict(parameter=name + '.weight', global_row=VOCAB_ID,
                                shard_rank=rank, shard_start=lo, shard_shape=list(value.shape),
                                shard_device=str(value.device), dtype=str(value.dtype),
                                shard_pointer=value.data_ptr(), sharded_state=str(p.sharded_state),
                                group_training_state=str(group._training_state),
                                full_storage_bytes=full.untyped_storage().nbytes() if full is not None else None,
                                all_gather_result_pending=group._all_gather_result is not None)
                            rows.append(row)
                            if not lo <= VOCAB_ID < lo + value.shape[0]:
                                row['not_owned_on_this_rank'] = True
                            elif value.device.type != 'cpu':
                                row['not_captured'] = 'Expected CPU shard unavailable; no additional GPU reads'
                            else:
                                row['buffer'] = value[VOCAB_ID - lo].detach().clone()
                except Exception as error:
                    errors.append(repr(error))
                output = self._diagnostic_async_head_output
                path = Path(output) / f'rank{self.rank}-postfailure-shard.pt'
                torch.save(rows, path)
                report = dict(rank=self.rank, unix=time.time(), source=__file__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_update_error=repr(failure) if failure is not None else None,
                    original_exception_preserved=True, errors=errors,
                    observations=[{k: v for k, v in r.items() if k != 'buffer'} for r in rows],
                    artifact=dict(path=str(path), bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    pre_update_exit_value_reads=0, added_GPU_calls=0,
                    added_synchronizations=0, extra_model_calls=0, production_changes=0)
                (Path(output) / f'rank{self.rank}-postfailure-shard.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(NativePostfailureShardObserved)


original_gemm_factory = gemm.make_observed_worker


def main():
    assert hashlib.sha256(Path(gemm.__file__).read_bytes()).hexdigest() == GEMM_SHA
    native_factory = gemm.make_observed_worker
    gemm.make_observed_worker = make_observed_worker
    try:
        gemm.main()
    finally:
        gemm.make_observed_worker = native_factory


if __name__ == '__main__':
    main()
