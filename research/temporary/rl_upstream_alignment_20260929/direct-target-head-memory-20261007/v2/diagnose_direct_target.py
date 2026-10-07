"""One real B4 through the original PrefixWorker/readout, exact head comparison.

Only the diagnostic call boundary is composed here. The existing v4 loader,
the actual producer, original finite seed and candidate row-block owner perform
all preparation and numerical work. No operands or checkpoints are exported.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
import time
import traceback


def _identity(owner):
    path = Path(inspect.getsourcefile(owner))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _configuration(out=None):
    root = Path(out) if out is not None else Path(__file__).parent
    path = root / 'head-memory-candidate.json'
    return json.loads(path.read_bytes()), dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _module(binding, name):
    path = Path(binding['path'])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != binding['sha256']:
        raise ValueError('Diagnostic source binding differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _base(out=None):
    config, _ = _configuration(out)
    return _module(config['base_diagnostic'], '_head_memory_original_v4_diagnostic')


def load_rows(path, rank):
    return _base(Path(path).parent).load_rows(path, rank)


def inspect_inputs(path):
    return _base(Path(path).parent).inspect_inputs(path)


def diagnose(runner, producer, out, save, *, cache_tensors=None):
    import psutil
    import torch

    out = Path(out)
    rank = torch.distributed.get_rank()
    config, configuration = _configuration(out)
    base = _module(config['base_diagnostic'], '_head_memory_original_v4_diagnostic')
    candidate_module = _module(config['candidate_answer'], '_head_memory_candidate_answer_' + str(os.getpid()))
    inputs = out / 'actual-direct-target-inputs.json'
    inspected = base.inspect_inputs(inputs)
    rows, _ = base.load_rows(inputs, rank)
    readout = producer.direct_readout
    if readout is None or readout.runner is not runner or readout.minibatch_size != 4 or len(rows) != 4:
        raise ValueError('Use the actual direct-target producer and four original rows')
    original_answer = runner.answer
    candidate_answer = candidate_module.FiniteAnswerOps(compiled=False)
    candidate_answer.seed = original_answer.seed
    candidate_answer.categorical_seed = original_answer.categorical_seed
    candidate_answer.dynamic_shapes = original_answer.dynamic_shapes
    body_globals = runner.attribute.__func__.__globals__
    original_log_probs = body_globals['selected_target_log_probs']
    candidate_log_probs = candidate_module.selected_target_log_probs
    process = psutil.Process()
    events = []
    answer_calls = 0
    log_prob_calls = 0

    def resources():
        memory = process.memory_full_info()
        free, total = torch.cuda.mem_get_info()
        return dict(observed_unix=time.time(), rss_bytes=memory.rss,
                    pss_bytes=memory.pss, physical_free_bytes=free,
                    physical_total_bytes=total,
                    torch_allocated_bytes=torch.cuda.memory_allocated(),
                    torch_reserved_bytes=torch.cuda.memory_reserved())

    def call(owner, *args, **kwargs):
        torch.cuda.synchronize()
        before = resources()
        started = time.perf_counter()
        value = owner(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return value, dict(seconds=elapsed, before=before, after=resources())

    def compare(actual, reference, path, fields):
        if isinstance(actual, torch.Tensor) and isinstance(reference, torch.Tensor):
            equal = torch.equal(actual, reference)
            item = dict(field=path, equal=equal, actual_shape=list(actual.shape),
                        reference_shape=list(reference.shape), actual_dtype=str(actual.dtype),
                        reference_dtype=str(reference.dtype), max_abs_difference=None)
            if actual.shape == reference.shape:
                maximum = 0.0
                # Diagnostic reduction only. Bounded temporaries avoid making
                # a full FP64 copy of the dense hidden-state result.
                for a, b in zip(actual.reshape(-1).split(262144),
                                reference.reshape(-1).split(262144)):
                    value = (a.double() - b.double()).abs().max().item() if a.numel() else 0.0
                    maximum = max(maximum, value) if value == value else float('nan')
                item['max_abs_difference'] = maximum
            fields.append(item)
            return equal
        if isinstance(actual, dict) and isinstance(reference, dict):
            keys_equal = actual.keys() == reference.keys()
            fields.append(dict(field=path, keys_equal=keys_equal))
            values_equal = [compare(actual[key], reference[key], path + '.' + str(key), fields)
                            for key in actual.keys() & reference.keys()]
            return keys_equal and all(values_equal)
        if isinstance(actual, (tuple, list)) and isinstance(reference, (tuple, list)):
            length_equal = len(actual) == len(reference)
            values_equal = [compare(a, b, path + '[' + str(i) + ']', fields)
                            for i, (a, b) in enumerate(zip(actual, reference))]
            return length_equal and all(values_equal)
        equal = actual == reference
        fields.append(dict(field=path, equal=bool(equal)))
        return bool(equal)

    def emit(event):
        events.append(event)
        print('[direct-target head memory] ' + json.dumps(event), flush=True)
        if not event['equal']:
            save('head_memory_exact_comparison_failed', inputs=inspected,
                 configuration=configuration, events=events,
                 operations=dict(original_B4_attribute_calls=1, rollout=0,
                                 optimizer=0, backward=0, checkpoint=0, operand_exports=0))
            raise AssertionError('Original head output differs from row-block candidate: ' + event['phase'])

    def answer(*args, **kwargs):
        nonlocal answer_calls
        answer_calls += 1
        logits, head, selection = args[:3]
        original, original_cost = call(original_answer, *args, **kwargs)
        candidate, candidate_cost = call(candidate_answer, *args, **kwargs)
        fields = []
        equal = compare(candidate, original, 'answer', fields)
        emit(dict(phase='finite_answer_exact_comparison', rank=rank, call=answer_calls,
                  input_shape=list(logits.shape), input_dtype=str(logits.dtype),
                  target_rows=len(selection.labels), vocabulary=head.out_features,
                  target_counts=list(selection.counts), equal=equal, fields=fields,
                  original=original_cost, candidate=candidate_cost,
                  timing_scope='Individual owner calls, excluding comparison and original native forward'))
        return candidate

    def log_probs(*args, **kwargs):
        nonlocal log_prob_calls
        log_prob_calls += 1
        original, original_cost = call(original_log_probs, *args, **kwargs)
        candidate, candidate_cost = call(candidate_log_probs, *args, **kwargs)
        fields = []
        equal = compare(candidate, original, 'selected_target_log_probs', fields)
        emit(dict(phase='selected_target_log_probs_exact_comparison', rank=rank,
                  call=log_prob_calls, input_shape=list(args[0].shape),
                  input_dtype=str(args[0].dtype), equal=equal, fields=fields,
                  original=original_cost, candidate=candidate_cost,
                  timing_scope='Individual owner calls, excluding comparison and original native forward'))
        return candidate

    save('head_memory_real_joint_inputs', inputs=inspected, configuration=configuration,
         owners=dict(producer=_identity(type(producer)), readout=_identity(type(readout)),
                     runner=_identity(runner.attribute), original_answer=_identity(type(original_answer)),
                     candidate_answer=_identity(candidate_module.FiniteAnswerOps),
                     original_log_probs=_identity(original_log_probs),
                     candidate_log_probs=_identity(candidate_log_probs)),
         exact_assertion_source='Original test_answer_boundary.py::test_default_text_target_keeps_original_finite_seed: torch.equal',
         operations=dict(rollout=0, optimizer=0, backward=0, checkpoint=0, operand_exports=0))
    runner.answer = answer
    body_globals['selected_target_log_probs'] = log_probs
    started = time.perf_counter()
    before = resources()
    try:
        readout.trajectories(rows)
        torch.cuda.synchronize()
        if readout.last_report['finite_trace_calls'] != 1:
            raise ValueError('The bounded real input must issue one original B4 attribution')
        save('head_memory_real_joint_complete', inputs=inspected, configuration=configuration,
             events=events, report=readout.last_report, seconds=time.perf_counter() - started,
             before=before, after=resources(),
             original_answer_calls=answer_calls, original_log_prob_calls=log_prob_calls,
             operations=dict(original_B4_attribute_calls=1, rollout=0, optimizer=0,
                             backward=0, checkpoint=0, operand_exports=0),
             numerical_scope='Exact original dtype head/seed/root comparison on the same real B4; original FA/FLA are unchanged',
             timing_scope='Complete readout includes duplicate diagnostic head computations and comparison, not a training speed measurement')
    except BaseException:
        save('head_memory_real_joint_failed', inputs=inspected, configuration=configuration,
             failure=traceback.format_exc(), events=events,
             before=before, after=resources(),
             operations=dict(original_B4_attribute_calls=1, rollout=0, optimizer=0,
                             backward=0, checkpoint=0, operand_exports=0))
        raise
    finally:
        runner.answer = original_answer
        body_globals['selected_target_log_probs'] = original_log_probs
