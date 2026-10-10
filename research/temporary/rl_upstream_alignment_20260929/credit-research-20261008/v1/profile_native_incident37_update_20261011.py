"""Profile the original incident replay's native update, without changing math.

Reuse the verified exception-capture wrapper and original e48ed002 replay.
Only the native actor update is profiled; old/ref and DT are unchanged. The
head observer records Python tensor metadata after the native head returns;
it does not read values, clone tensors, synchronize, or retain tensor objects.
Profiling can perturb timing. A finite profiled replay is not a repair.
"""
import functools
import hashlib
import json
from pathlib import Path
import time

import replay_native_incident37_exception_capture_20261011 as capture


CAPTURE_SHA = '3f4942a9c5135eb1d52ae1393d3fba82f46fa28cdf99509d5f2606da8a8e4f23'


def make_profiled_worker(native_factory):
    import ray
    actor = native_factory()
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_replay = native_class.replay_incident
    native_update = native_class.update_actor

    class NativeUpdateProfiled(native_class):
        @functools.wraps(native_replay)
        def replay_incident(self, output):
            self._diagnostic_profile_output = Path(output)
            return native_replay(self, output)

        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from verl.utils.experimental import torch_functional as head
            from capture_native_loss_backward_20261010 import observe_call

            output = self._diagnostic_profile_output
            native_head = head.FusedLinearForPPO.forward
            rows = []

            def metadata(tensor):
                return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                            dtype=str(tensor.dtype), device=str(tensor.device),
                            data_ptr=tensor.data_ptr(),
                            storage_ptr=tensor.untyped_storage().data_ptr(),
                            storage_bytes=tensor.untyped_storage().nbytes(),
                            storage_offset=tensor.storage_offset(),
                            requires_grad=tensor.requires_grad)

            def observed(result, arguments):
                rows.append(dict(unix_ns=time.time_ns(),
                    current_stream=torch.cuda.current_stream().cuda_stream,
                    head_call=len(rows),
                    inputs={key:metadata(arguments[key]) for key in
                            ['hidden_states', 'vocab_weights', 'input_ids']},
                    outputs=[metadata(value) for value in result]))

            def observation_error(error):
                rows.append(dict(unix_ns=time.time_ns(), error=repr(error)))

            head.FusedLinearForPPO.forward = observe_call(
                native_head, lambda: True, observed, observation_error)
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=True, with_stack=False)
            try:
                with profiler:
                    return native_update(self, data)
            finally:
                head.FusedLinearForPPO.forward = native_head
                report = dict(rank=self.rank, unix=time.time(),
                              source=__file__,
                              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              head_calls=rows, tensor_value_observations=0,
                              original_head_restored=True,
                              profile_scope='Original native update only')
                trace = output / f'rank{self.rank}-native-update-trace.json'
                try:
                    profiler.export_chrome_trace(str(trace))
                    report['trace'] = dict(path=str(trace), bytes=trace.stat().st_size,
                        sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
                except Exception as error:
                    report['trace_error'] = repr(error)
                (output/f'rank{self.rank}-native-update-profile.json').write_text(
                    json.dumps(report, indent=2)+'\n')

    return ray.remote(NativeUpdateProfiled)


def main():
    assert hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest() == CAPTURE_SHA
    native_factory = capture.original.make_worker
    capture.original.make_worker = lambda: make_profiled_worker(native_factory)
    try:
        capture.main()
    finally:
        capture.original.make_worker = native_factory


if __name__ == '__main__':
    main()
