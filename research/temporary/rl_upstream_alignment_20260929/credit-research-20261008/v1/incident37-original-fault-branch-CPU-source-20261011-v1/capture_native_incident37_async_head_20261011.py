"""Capture the second native head result without a pre-loss CPU value read.

Only queue native non-blocking copies to pinned CPU buffers on the current
stream. Read them after the original replay returns or raises. This diagnostic
can still perturb timing; a finite run is not a repair. No production changes.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import replay_native_incident37_exception_capture_20261011 as capture


CAPTURE_SHA = '06c0b6f5a0c8fc24e5eeecfd3696acda0186a8512e63a4f24540a63350b8e022'


def make_observed_worker(native_factory):
    import ray
    actor = native_factory()
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_replay = native_class.replay_incident
    native_update = native_class.update_actor

    class NativeAsyncHeadObserved(native_class):
        @functools.wraps(native_replay)
        def replay_incident(self, output):
            self._diagnostic_async_head_output = Path(output)
            return native_replay(self, output)

        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from verl.utils.experimental import torch_functional as head
            from capture_native_loss_backward_20261010 import observe_call

            output = self._diagnostic_async_head_output
            native_head = head.FusedLinearForPPO.forward
            pending, errors = [], []
            call_index = 0

            def observed(result, arguments):
                nonlocal call_index
                index = call_index
                call_index += 1
                if index != 1:
                    return
                stream = torch.cuda.current_stream()
                buffers = []
                with torch.no_grad():
                    for value in result:
                        buffer = torch.empty_like(value, device='cpu', pin_memory=True)
                        buffer.copy_(value.detach(), non_blocking=True)
                        buffers.append(buffer)
                event = torch.cuda.Event()
                event.record(stream)
                pending.append(dict(index=index, unix_ns=time.time_ns(), event=event,
                                    buffers=buffers, stream=stream.cuda_stream,
                                    head_input_shapes={key:list(arguments[key].shape) for key in
                                        ['hidden_states', 'vocab_weights', 'input_ids']},
                                    head_output_shapes=[list(value.shape) for value in result],
                                    copied_bytes=sum(b.numel() * b.element_size() for b in buffers)))

            head.FusedLinearForPPO.forward = observe_call(
                native_head, lambda: True, observed,
                lambda error: errors.append(repr(error)))
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                head.FusedLinearForPPO.forward = native_head
                report = dict(rank=self.rank, unix=time.time(),
                              source=__file__,
                              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              native_update_error=repr(failure) if failure is not None else None,
                              head_calls=call_index, errors=errors, copies=[],
                              pre_loss_cpu_value_reads=0, pre_loss_explicit_synchronizations=0,
                              original_head_restored=True)
                for item in pending:
                    # Native replay has already returned or failed. Reading
                    # these CPU buffers earlier would violate copy_ semantics.
                    item['event'].synchronize()
                    path = Path(output) / f'rank{self.rank}-async-head-return-microbatch{item["index"]}.pt'
                    metadata = {k:v for k,v in item.items() if k not in ['event', 'buffers']}
                    torch.save(dict(log_probs=item['buffers'][0], entropy=item['buffers'][1],
                                    metadata=metadata), path)
                    report['copies'].append(dict(**metadata, path=str(path),
                        bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                (Path(output) / f'rank{self.rank}-async-head-capture.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(NativeAsyncHeadObserved)


def main():
    assert hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest() == CAPTURE_SHA
    native_factory = capture.original.make_worker
    capture.original.make_worker = lambda: make_observed_worker(native_factory)
    try:
        capture.main()
    finally:
        capture.original.make_worker = native_factory


if __name__ == '__main__':
    main()
