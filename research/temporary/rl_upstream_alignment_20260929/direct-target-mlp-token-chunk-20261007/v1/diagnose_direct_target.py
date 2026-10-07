"""Same-capture original MLP versus token-chunk composition on real rows.

The existing native PrefixWorker, row loader, model, producer, targets and
readout remain the owners. This diagnostic temporarily disables native-prefix
reuse so the saved real rows exercise more than one 2048-token MLP chunk.
It is neither a formal-throughput run nor a 32768-token capacity certificate.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback

from lifetime_diagnostic_owner import _module, inspect_inputs, load_rows


def diagnose(runner, producer, out, save, *, cache_tensors=None):
    import psutil
    import torch

    out = Path(out)
    config = json.loads((out / 'lifetime-comparison.json').read_bytes())
    adapter = _module(config['chunk_owner'], '_mlp_token_chunk_adapter')
    chunk_size = config['token_chunk_size']
    original_mlp = runner.boundaries.mlp
    original_reuse_native_prefix = runner.reuse_native_prefix
    rank = torch.distributed.get_rank()
    rows, _ = load_rows(out / 'actual-direct-target-inputs.json', rank)
    inspected = inspect_inputs(out / 'actual-direct-target-inputs.json')
    assert len(rows) == 4 and producer.direct_readout.minibatch_size == 4
    events = []

    def identity(owner):
        path = Path(inspect.getsourcefile(owner))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def resources():
        return dict(observed_unix=time.time(),
                    pss_bytes=psutil.Process().memory_full_info().pss,
                    cuda_reported_free_bytes=torch.cuda.mem_get_info()[0],
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved())

    def compare(*args):
        torch.cuda.synchronize()
        original_before = resources()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        reference = original_mlp(*args)
        torch.cuda.synchronize()
        original_seconds = time.perf_counter() - started
        original_peak = torch.cuda.max_memory_allocated()
        chunk_calls = []

        def counted_original(*chunk_args):
            chunk_calls.append(dict(
                batch=int(chunk_args[0].shape[0]),
                tokens=int(chunk_args[0].shape[1]),
                shapes=[list(value.shape) for value in chunk_args[:7]],
                dtypes=[str(value.dtype) for value in chunk_args[:7]],
                strides=[list(value.stride()) for value in chunk_args[:7]]))
            return original_mlp(*chunk_args)

        chunked = adapter._chunk_mlp_tokens(counted_original, chunk_size)
        candidate_before = resources()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actual = chunked(*args)
        torch.cuda.synchronize()
        candidate_seconds = time.perf_counter() - started
        candidate_peak = torch.cuda.max_memory_allocated()
        event = dict(phase='same_capture_mlp_token_chunk_comparison', rank=rank,
                     call=len(events) + 1, input_shapes=[list(v.shape) for v in args[:7]],
                     shape=list(actual.shape), reference_shape=list(reference.shape),
                     dtype=str(actual.dtype), reference_dtype=str(reference.dtype),
                     original_seconds=original_seconds, candidate_seconds=candidate_seconds,
                     original_before=original_before, candidate_before=candidate_before,
                     original_peak_allocated_bytes=original_peak,
                     candidate_peak_allocated_bytes=candidate_peak,
                     retained_reference_output_bytes=reference.numel() * reference.element_size(),
                     chunk_calls=chunk_calls, chunk_count=len(chunk_calls),
                     token_chunk_size=chunk_size, comparison_passed=False)
        events.append(event)
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype
        equal = True
        max_abs_difference = 0.0
        # Only the comparison is split here. The numerical assertion uses
        # unmodified Torch dtype defaults; no new/looser tolerance is supplied.
        for a, b in zip(actual.reshape(-1).split(262144),
                        reference.reshape(-1).split(262144)):
            equal = torch.equal(a, b) and equal
            torch.testing.assert_close(a, b)
            max_abs_difference = max(max_abs_difference, (a - b).abs().max().item())
        del a, b, reference
        event.update(bitwise_equal=equal, max_abs_difference=max_abs_difference,
                     comparison_passed=True, after=resources())
        save('mlp_token_chunk_comparison_progress', latest=event,
             compared_mlp_calls=len(events))
        return actual

    save('mlp_token_chunk_real_input_start', inputs=inspected,
         owners=dict(runner=identity(runner.attribute), original_mlp=identity(original_mlp),
                     chunk_adapter=identity(adapter._chunk_mlp_tokens)),
         assertion='Unmodified torch.testing.assert_close dtype defaults on same-capture MLP outputs. Shape and dtype checked first. Not whole-chain FA/FLA or PPO tolerance certification.',
         scope='Saved real B8 response rows, original targets/rewards/readout; diagnostic-only full real-context replay, not formal throughput or 32k capacity.',
         runtime_override=dict(reuse_native_prefix_before=original_reuse_native_prefix,
                               reuse_native_prefix_during=False, restored_in_finally=True,
                               token_chunk_size=chunk_size),
         operations=dict(original_B4_attribute_calls_per_rank=1, rollout=0,
                         optimizer=0, actor_backward=0, checkpoint=0,
                         duplicate_mlp_owner_calls_for_comparison=True),
         memory_scope='Per-operation allocator peaks; candidate runs while retaining reference output. These are not standalone/full physical VRAM peaks. CUDA reported free is not mx-smi physical usage.')
    runner.boundaries.mlp = compare
    runner.reuse_native_prefix = False
    started = time.perf_counter()
    try:
        producer.direct_readout.trajectories(rows)
        torch.cuda.synchronize()
        assert producer.direct_readout.last_report['finite_trace_calls'] == 1
        save('mlp_token_chunk_real_input_complete', events=events,
             report=producer.direct_readout.last_report,
             all_comparisons_passed=all(event['comparison_passed'] for event in events),
             all_bitwise_equal=all(event.get('bitwise_equal', False) for event in events),
             observed_multi_chunk_calls=sum(event['chunk_count'] > 1 for event in events),
             observed_nondivisible_calls=sum(event['input_shapes'][0][1] % chunk_size != 0
                                            for event in events),
             seconds=time.perf_counter() - started, resources=resources(),
             cost_scope='Includes original full MLP plus chunked MLP, compilation and numerical comparison, with prefix reuse disabled. Not formal throughput or full physical peak memory.')
    except BaseException:
        save('mlp_token_chunk_real_input_failed', failure=traceback.format_exc(),
             events=events, resources=resources())
        raise
    finally:
        runner.boundaries.mlp = original_mlp
        runner.reuse_native_prefix = original_reuse_native_prefix
