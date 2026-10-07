"""Reuse original author-curve diagnostic, one bound existing rule per DP rank.

Only the bound real input attribution artifact differs. The original native
scorer, literal-ID adapter, cumulative deletion, k=20 and RISE/MAS are called
unchanged. No DT, optimizer, sampling or production rule deployment occurs.
"""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import threading
import time

import inspect_action_curve as owner

def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class CurveComparisonWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output):
            manifest = json.loads((Path(output).parent / 'curve-inputs.json').read_bytes())
            binding = manifest['ranks'][str(self.rank)]
            previous = owner.CASES['appworld']
            owner.CASES['appworld'] = binding['case']
            # Ray workers import the staged owner module independently; the
            # driver's main-only replacement is not the worker's constructor.
            original_make_worker = owner.make_worker
            assert original_make_worker.__module__ == 'inspect_action_curve'
            original_class = original_make_worker().__ray_metadata__.modified_class
            try:
                result = original_class.inspect_action_curve(self, source_path, output)
                path = Path(output) / f'rank{self.rank}.json'
                report = json.loads(path.read_bytes())
                report['profile_comparison_binding'] = binding
                report['comparison_scope'] = __doc__
                report['original_curve_diagnostic'] = dict(
                    path=inspect.getsourcefile(original_make_worker),
                    sha256=hashlib.sha256(Path(inspect.getsourcefile(original_make_worker)).read_bytes()).hexdigest())
                path.write_text(json.dumps(report, indent=2) + '\n')
                return result
            finally:
                owner.CASES['appworld'] = previous

    return CurveComparisonWorker


if __name__ == '__main__':
    # The existing main owns configuration, native actor creation and teardown.
    # Sampling physical mx-smi is observation only, not a memory policy.
    import sys
    output = Path(sys.argv[sys.argv.index('--output') + 1]).parent
    stopped = threading.Event()

    def sample():
        with (output / 'curve-physical-mx-smi.jsonl').open('a', buffering=1) as stream:
            while not stopped.is_set():
                r = subprocess.run(['mx-smi'], capture_output=True, text=True, timeout=20)
                stream.write(json.dumps(dict(unix=time.time(), returncode=r.returncode,
                                            stdout=r.stdout, stderr=r.stderr)) + '\n')
                stopped.wait(2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    owner.make_worker = make_worker
    try:
        owner.main()
    finally:
        stopped.set()
        thread.join(timeout=25)
