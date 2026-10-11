"""Record native allocator ownership during the saved incident update.

Compose the existing GEMM observer unchanged. PyTorch owns allocation history
and snapshot serialization; no model, collective, numerical or lifetime
operation is replaced. No extra tensor reads or GPU copies before update exit.
This is diagnostic-only: observation can change timing and is not a repair.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import capture_native_incident37_GEMM_rows_20261011 as gemm


GEMM_SHA = '35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8'
MEMORY_OWNER_SHA = '3afde64252800d6ce537ff62d471d1c60f9c07ea9c0022074db299d90d1f865d'


def make_observed_worker(native_factory):
    import ray
    actor = original_gemm_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class NativeAllocatorObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import inspect
            import torch

            owner = Path(inspect.getsourcefile(torch.cuda.memory._snapshot))
            assert hashlib.sha256(owner.read_bytes()).hexdigest() == MEMORY_OWNER_SHA
            output = Path(self._diagnostic_async_head_output)
            report = dict(rank=self.rank, source=__file__,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                memory_owner=str(owner), memory_owner_sha256=MEMORY_OWNER_SHA,
                started_unix=time.time(), max_entries=60000,
                context='all', stacks='python', production_changes=0,
                added_model_calls=0, added_tensor_copies=0,
                added_explicit_synchronizations=0, errors=[])
            # Existing native allocator API records alloc/free metadata and
            # stacks. It does not add a model call or change tensor lifetimes.
            torch.cuda.memory._record_memory_history(
                enabled='all', context='all', stacks='python', max_entries=60000)
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                report['native_update_error'] = repr(failure) if failure is not None else None
                report['original_exception_preserved'] = True
                try:
                    path = output / f'rank{self.rank}-native-allocator.pickle'
                    torch.cuda.memory._dump_snapshot(str(path))
                    report['snapshot'] = dict(path=str(path), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                except Exception as error:
                    report['errors'].append(repr(error))
                finally:
                    try:
                        torch.cuda.memory._record_memory_history(enabled=None)
                    except Exception as error:
                        report['errors'].append('disable memory history: ' + repr(error))
                report['finished_unix'] = time.time()
                (output / f'rank{self.rank}-native-allocator.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(NativeAllocatorObserved)


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
