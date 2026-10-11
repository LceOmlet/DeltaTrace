"""Record original incident allocator events without per-event stack capture.

The earlier 60k ring retained only later microbatches. Keep 1M native events
so the actual selected second-head interval can be checked afterwards. This
only invokes PyTorch's existing recorder/snapshot APIs; no tensor value read,
extra model operation, or training change is introduced.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import capture_native_incident37_GEMM_rows_20261011 as gemm


GEMM_SHA = '35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8'
MEMORY_SHA = '3afde64252800d6ce537ff62d471d1c60f9c07ea9c0022074db299d90d1f865d'
original_factory = gemm.make_observed_worker


def make_worker(native_factory):
    import ray
    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class AllocatorWindowObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import inspect
            import torch
            owner = Path(inspect.getsourcefile(torch.cuda.memory._record_memory_history))
            assert hashlib.sha256(owner.read_bytes()).hexdigest() == MEMORY_SHA
            output = Path(self._diagnostic_async_head_output)
            report = dict(rank=self.rank, started_unix=time.time(), context=None,
                          max_entries=1000000, owner=str(owner), owner_sha256=MEMORY_SHA,
                          source=str(Path(__file__)),
                          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          production_changes=0, added_tensor_copies=0,
                          added_model_calls=0, errors=[])
            torch.cuda.memory._record_memory_history(enabled='all', context=None,
                                                     max_entries=report['max_entries'])
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                report['native_update_error'] = repr(failure) if failure is not None else None
                try:
                    path = output / f'rank{self.rank}-native-allocator.pickle'
                    torch.cuda.memory._dump_snapshot(str(path))
                    report['snapshot'] = dict(path=str(path), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                except Exception as error:
                    report['errors'].append(repr(error))
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)
                report['finished_unix'] = time.time()
                (output / f'rank{self.rank}-allocator-window.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(AllocatorWindowObserved)


def main():
    assert hashlib.sha256(Path(gemm.__file__).read_bytes()).hexdigest() == GEMM_SHA
    gemm.make_observed_worker = make_worker
    try:
        gemm.main()
    finally:
        gemm.make_observed_worker = original_factory


if __name__ == '__main__':
    main()
