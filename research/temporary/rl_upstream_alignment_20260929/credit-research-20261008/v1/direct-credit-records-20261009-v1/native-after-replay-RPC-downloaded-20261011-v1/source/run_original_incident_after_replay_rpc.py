"""Prepared diagnostic: original replay, then official VERL CPU observer RPC.

The original replay and worker classes remain untouched. The sole driver seam
is its final ray.shutdown call, after native reports have already been written.
This is not a production launcher or an accepted repair.
"""
from pathlib import Path
import hashlib
import json
import sys
import traceback

import ray
import run_original_incident_native_recorder as driver
from observe_CPU_source_after_replay_rpc import read_CPU_source


DRIVER_SHA = '15bf3d5443c1db47fb6e907f47abce288ef6f55ae3e7cef1a0ca0d54191d4d1b'
REPLAY_SHA = 'e48ed0021391b2d6c1473714d9e90ab6d44bfed6026697572dc8f0596e40ccb9'
WORKER_SHA = '4863a3d69016ab7b2ab97053dfad7bb1c4bdaacccdf9404696b2a8386f75f17b'
GROUP_SHA = 'bcdb6ca4e7e9e9392338157012e988dc39e12a85f4de540b6e66ffe3fd914ef4'
NATIVE_OBSERVER_SHA = '4e6e437119e1a9f1c232cef3d01f9eca3d1a8b4cb6c2411ac2de097698cad4c2'


def main():
    import inspect
    import importlib.util
    from verl.single_controller.base.worker import Worker
    from verl.single_controller.ray import RayWorkerGroup

    assert hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest() == DRIVER_SHA
    for cls, expected in ((Worker, WORKER_SHA), (RayWorkerGroup, GROUP_SHA)):
        assert hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() == expected
    # Select the failure-capturing v1 recorder, without the v2/v3 CPU callback.
    # Merely locate the module; the driver imports no observer on the worker.
    spec = importlib.util.find_spec('dump_native_collectives_after_report')
    assert spec and hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() == NATIVE_OBSERVER_SHA
    original_shutdown = ray.shutdown

    def shutdown_after_observation(*args, **kwargs):
        frame = sys._getframe(1)
        output = None
        try:
            caller = Path(frame.f_code.co_filename)
            if frame.f_code.co_name == 'main' and caller.is_file() and hashlib.sha256(caller.read_bytes()).hexdigest() == REPLAY_SHA:
                group = frame.f_locals.get('group')
                options = frame.f_locals.get('args')
                if group is not None and options is not None:
                    output = Path(options.output)
                    results = group.execute_all_sync('execute_with_func_generator', read_CPU_source, str(output))
                    (output / 'after-replay-CPU-source-RPC.json').write_bytes(
                        (json.dumps({'results': results, 'production_changes': 0,
                                     'pre_terminal_worker_changes': 0}, indent=2) + '\n').encode())
        except Exception:
            # An unavailable diagnostic must not hide the original replay error.
            diagnostic_error = traceback.format_exc()
            if output is not None:
                try:
                    (output / 'after-replay-CPU-source-RPC-error.txt').write_text(diagnostic_error)
                except OSError:
                    print(diagnostic_error, file=sys.stderr)
        finally:
            del frame
            original_shutdown(*args, **kwargs)

    ray.shutdown = shutdown_after_observation
    try:
        driver.main()
    finally:
        ray.shutdown = original_shutdown


if __name__ == '__main__':
    main()
