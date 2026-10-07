"""Compare lifetime-only GDN owners on the same actual native captures.

The existing PrefixWorker, v4 real-ID loader, producer and readout own all
computation. The comparison calls the two original owner functions; no finite
rule or model forward is reimplemented. Inputs are the saved real B8 response
rows, not the lost terminal batch and not a 32k capacity certificate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


def _module(binding, name):
    path = Path(binding['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == binding['sha256'], path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _configuration(out):
    path = Path(out) / 'lifetime-comparison.json'
    return json.loads(path.read_bytes())


def _base(out):
    return _module(_configuration(out)['base_diagnostic'], '_lifetime_real_input_loader')


def load_rows(path, rank):
    return _base(Path(path).parent).load_rows(path, rank)


def inspect_inputs(path):
    return _base(Path(path).parent).inspect_inputs(path)


def diagnose(runner, producer, out, save, *, cache_tensors=None):
    import psutil
    import torch

    out = Path(out)
    config = _configuration(out)
    baseline = _module(config['baseline_gdn'], '_lifetime_baseline_gdn')
    globals_ = runner.attribute.__func__.__globals__
    candidate = globals_['gdn_finite_pullback']
    decoder = globals_['decoder_finite_pullback']
    rank = torch.distributed.get_rank()
    rows, _ = load_rows(out / 'actual-direct-target-inputs.json', rank)
    inspected = inspect_inputs(out / 'actual-direct-target-inputs.json')
    assert len(rows) == 4 and producer.direct_readout.minibatch_size == 4
    assert not runner.offload_replay_mixer
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

    def compare(module, values, endpoints, *args, **kwargs):
        assert kwargs['consume_captures'] is True
        assert kwargs['offload_endpoints'] is False
        original_kwargs = {k: v for k, v in kwargs.items() if k != 'consume_captures'}
        torch.cuda.synchronize()
        started = time.perf_counter()
        reference, reference_terms = baseline.gdn_finite_pullback(
            module, dict(values), dict(endpoints), *args, **original_kwargs)
        torch.cuda.synchronize()
        original_seconds = time.perf_counter() - started
        assert not reference_terms
        before = resources()
        started = time.perf_counter()
        actual, terms = candidate(module, values, endpoints, *args, **kwargs)
        torch.cuda.synchronize()
        candidate_seconds = time.perf_counter() - started
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype
        equal = True
        max_abs_difference = 0.0
        # Same operands and arithmetic: no changed tolerance. Chunking only
        # bounds the diagnostic comparison's temporary allocation.
        for a, b in zip(actual.reshape(-1).split(262144),
                        reference.reshape(-1).split(262144)):
            equal = torch.equal(a, b) and equal
            torch.testing.assert_close(a, b)
            max_abs_difference = max(max_abs_difference,
                                     (a - b).abs().max().item())
        event = dict(phase='same_capture_gdn_lifetime_comparison', rank=rank,
                     call=len(events) + 1, shape=list(actual.shape),
                     dtype=str(actual.dtype), bitwise_equal=equal,
                     max_abs_difference=max_abs_difference,
                     original_seconds=original_seconds,
                     candidate_seconds=candidate_seconds,
                     remaining_values=sorted(values),
                     remaining_endpoints=sorted(endpoints), before=before,
                     after=resources())
        events.append(event)
        save('gpu_lifetime_comparison_progress', latest=event,
             compared_gdn_calls=len(events))
        return actual, terms

    def observe_decoder(layer, captures, *args, **kwargs):
        assert kwargs['consume_captures'] is True
        result = decoder(layer, captures, *args, **kwargs)
        assert all(name not in captures for name in
                   ('gate_output', 'up_output', 'silu_output', 'post_norm_input'))
        assert 'input_norm_input' in captures
        return result

    save('gpu_lifetime_real_input_start', inputs=inspected,
         owners=dict(runner=identity(runner.attribute), candidate_gdn=identity(candidate),
                     baseline_gdn=identity(baseline.gdn_finite_pullback),
                     decoder=identity(decoder)),
         assertion='Unmodified torch.testing.assert_close dtype defaults on actual same-capture outputs; bitwise equality also reported. FA/FLA owner assertions and kernels unchanged.',
         scope='Actual saved B8 current-response rows; not terminal joint-target or 32k capacity proof',
         operations=dict(original_B4_attribute_calls_per_rank=1, rollout=0,
                         optimizer=0, actor_backward=0, checkpoint=0,
                         original_gdn_reverse_and_conv_autograd_retained=True))
    globals_['gdn_finite_pullback'] = compare
    globals_['decoder_finite_pullback'] = observe_decoder
    started = time.perf_counter()
    try:
        producer.direct_readout.trajectories(rows)
        torch.cuda.synchronize()
        assert len(events) == 24
        assert producer.direct_readout.last_report['finite_trace_calls'] == 1
        save('gpu_lifetime_real_input_complete', events=events,
             report=producer.direct_readout.last_report,
             all_bitwise_equal=all(event['bitwise_equal'] for event in events),
             seconds=time.perf_counter() - started, resources=resources(),
             cost_scope='Includes duplicate baseline GDN calls and comparisons; not a formal speed or peak-memory measurement')
    except BaseException:
        save('gpu_lifetime_real_input_failed', failure=traceback.format_exc(),
             events=events, resources=resources())
        raise
    finally:
        globals_['gdn_finite_pullback'] = candidate
        globals_['decoder_finite_pullback'] = decoder
