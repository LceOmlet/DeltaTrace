"""Observe existing compiled head GEMM rows; do not replace any calculation.

Selection is fixed from the preceding failed incident replay. This is incident
localization, not a representative attribution test or a production patch.
All CPU reads/waits occur after native update_actor has returned or raised.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import capture_native_incident37_async_head_20261011 as async_head


ASYNC_SOURCE_SHA = '782db0e8548f7cfb12bcf9c24271400da895b8bcb9e008eb3813f9014a0e9cc2'
SELECTION_SHA = '9a82a6912a18e44cf11cca67bcc217ffbcdf6d16dbc8adc71bd487f461c5cd1c'


def make_observed_worker(native_factory):
    import ray
    actor = original_async_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class NativeGEMMRowsObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from torch._inductor.select_algorithm import extern_kernels
            from verl.utils.experimental import torch_functional as head

            output = self._diagnostic_async_head_output
            selection_path = output.parent / 'selection.json'
            selection_bytes = selection_path.read_bytes()
            assert hashlib.sha256(selection_bytes).hexdigest() == SELECTION_SHA
            selection = json.loads(selection_bytes)['ranks'][self.rank]
            native_head = head.FusedLinearForPPO.forward
            signature = inspect.signature(native_head)
            calls, pending, errors = 0, [], []

            @functools.wraps(native_head)
            def observed_head(*args, **kwargs):
                nonlocal calls
                index = calls
                calls += 1
                if index != 1:
                    return native_head(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                h = bound.arguments['hidden_states']
                w = bound.arguments['vocab_weights']
                ids = bound.arguments['input_ids']
                assert list(ids.shape) == selection['head_shape']
                assert h.is_contiguous() and ids.is_contiguous()
                count, vocab = len(selection['rows']), selection['vocab']
                buffers = dict(
                    logits=torch.empty((count, vocab), dtype=h.dtype, device='cpu', pin_memory=True),
                    hidden=torch.empty((count, h.shape[-1]), dtype=h.dtype, device='cpu', pin_memory=True),
                    target_weights=torch.empty((count, w.shape[-1]), dtype=w.dtype, device='cpu', pin_memory=True),
                    token_ids=torch.empty(count, dtype=ids.dtype, device='cpu', pin_memory=True))
                ids_flat = ids.view(-1)
                native_mm = extern_kernels.mm
                row_start, mm_calls, copied = 0, [], []

                @functools.wraps(native_mm)
                def observed_mm(left, right, *mm_args, **mm_kwargs):
                    nonlocal row_start
                    # Generated Inductor wrapper calls this exact native mm.
                    result = native_mm(left, right, *mm_args, **mm_kwargs)
                    value = mm_kwargs.get('out', result)
                    if (left.ndim != 2 or right.ndim != 2 or value.ndim != 2
                            or right.shape != (selection['hidden_dim'], vocab)):
                        errors.append('Unexpected mm shape inside the selected native head')
                        return result
                    stop = row_start + left.shape[0]
                    metadata = dict(start=row_start, stop=stop,
                        left_shape=list(left.shape), right_shape=list(right.shape),
                        output_shape=list(value.shape), dtype=str(value.dtype),
                        left_pointer=left.data_ptr(), right_pointer=right.data_ptr(),
                        output_pointer=value.data_ptr(), stream=torch.cuda.current_stream().cuda_stream)
                    mm_calls.append(metadata)
                    try:
                        with torch.no_grad():
                            for j, row in enumerate(selection['rows']):
                                position = row['flat_position']
                                if row_start <= position < stop:
                                    local = position - row_start
                                    token = row['expected_token_id']
                                    # Views and native D2H copies only; no extra
                                    # GEMM, normalization or GPU index tensor.
                                    buffers['logits'][j].copy_(value[local].detach(), non_blocking=True)
                                    buffers['hidden'][j].copy_(left[local].detach(), non_blocking=True)
                                    buffers['target_weights'][j].copy_(right[:, token].detach(), non_blocking=True)
                                    buffers['token_ids'][j].copy_(ids_flat[position].detach(), non_blocking=True)
                                    copied.append(j)
                    except Exception as error:
                        errors.append(repr(error))
                    row_start = stop
                    return result

                extern_kernels.mm = observed_mm
                try:
                    return native_head(*args, **kwargs)
                finally:
                    extern_kernels.mm = native_mm
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream())
                    pending.append(dict(index=index, buffers=buffers, event=event,
                        mm_calls=mm_calls, copied_indices=copied, native_rows_seen=row_start,
                        expected_rows=ids.numel(), temperature=bound.arguments['temperature'],
                        selection=selection))

            head.FusedLinearForPPO.forward = observed_head
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                head.FusedLinearForPPO.forward = native_head
                report = dict(rank=self.rank, unix=time.time(), source=__file__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_update_error=repr(failure) if failure is not None else None,
                    head_calls=calls, errors=errors, captures=[],
                    pre_loss_cpu_value_reads=0, pre_loss_explicit_synchronizations=0,
                    extra_GEMM_calls=0, original_head_restored=True)
                for item in pending:
                    item['event'].synchronize()
                    metadata = {k:v for k,v in item.items() if k not in ['event', 'buffers']}
                    path = output / f'rank{self.rank}-native-GEMM-rows.pt'
                    torch.save(dict(**item['buffers'], metadata=metadata), path)
                    report['captures'].append(dict(path=str(path), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        copied_rows=len(item['copied_indices']),
                        planned_rows=len(selection['rows']), native_mm_calls=len(item['mm_calls']),
                        native_rows_seen=item['native_rows_seen'], expected_rows=item['expected_rows']))
                (output / f'rank{self.rank}-native-GEMM-capture.json').write_text(
                    json.dumps(report, indent=2) + '\n')

    return ray.remote(NativeGEMMRowsObserved)


original_async_factory = async_head.make_observed_worker


def main():
    assert hashlib.sha256(Path(async_head.__file__).read_bytes()).hexdigest() == ASYNC_SOURCE_SHA
    async_head.make_observed_worker = make_observed_worker
    try:
        async_head.main()
    finally:
        async_head.make_observed_worker = original_async_factory


if __name__ == '__main__':
    main()
