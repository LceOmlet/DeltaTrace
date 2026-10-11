"""Test the single owner lookup fix in the original failing stage replay."""
import functools
import hashlib
import json
from pathlib import Path
import time

import capture_root_native_stages as stages
import patch_owner_event_wait

original_factory = stages.make_worker


def make_worker(native_factory):
    import ray
    actor = original_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class NativeOwnerFixCandidate(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            receipt, restore = patch_owner_event_wait.apply()
            try:
                return native_update(self, data)
            finally:
                restore()
                receipt.update(rank=self.rank, unix=time.time(), restored_after_test=True,
                    source=patch_owner_event_wait.__file__,
                    source_SHA256=hashlib.sha256(Path(patch_owner_event_wait.__file__).read_bytes()).hexdigest(),
                    accepted_production_deployment=False)
                output = Path(self._diagnostic_async_head_output)
                (output / f'rank{self.rank}-owner-fix-override.json').write_text(json.dumps(receipt, indent=2) + '\n')

    return ray.remote(NativeOwnerFixCandidate)


def main():
    assert hashlib.sha256(Path(stages.__file__).read_bytes()).hexdigest() == '5a4d777b1cfd34df9e559a2e8d4e040303281a7539bc3f14fa1bb2789b902dde'
    stages.make_worker = make_worker
    try:
        stages.main()
    finally:
        stages.make_worker = original_factory


if __name__ == '__main__':
    main()
