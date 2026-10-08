"""Isolate the preserved runner's GDN callback; no alternative runner logic."""
import copy
from functools import partial
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_candidate(original, spec, environment):
    for item in spec['files']:
        assert sha(item['path']) == item['sha256']
    runner_path = inspect.getsourcefile(original.attribute)
    original_globals = original.attribute.__func__.__globals__
    native_gdn = original_globals['gdn_finite_pullback']
    assert sha(inspect.getsourcefile(native_gdn)) == spec['baseline_gdn_sha256']
    # Import the unchanged, actual runner under a private module identity.
    # Only this private copy binds the new default-inert owner callback.
    previous = {name: sys.modules.get(name) for name in
                ('finite_fla_gpu', 'qwen35_gdn_finite', 'conditional_gdn_context')}
    try:
        fla = load('finite_fla_gpu', spec['fla'])
        gdn = load('qwen35_gdn_finite', spec['gdn'])
        context = load('conditional_gdn_context', spec['context'])
        runner = load('research_conditional_gdn_runner', runner_path)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    runner.NativeGDNCapture = original_globals['NativeGDNCapture']
    layers = original.model.model.language_model.layers
    indices = {id(layer.linear_attn): i for i, layer in enumerate(layers)
               if layer.block_type == 'linear_attention'}
    output = Path(spec['phase_directory'])
    output.mkdir(exist_ok=True)
    stream = (output / ('gdn-context-pid'+str(os.getpid())+'.jsonl')).open('a', buffering=1)
    sequence = 0

    def event(phase, **values):
        stream.write(json.dumps(dict(phase=phase, unix=time.time(),
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            sequence=sequence, **values))+'\n')

    def callback(module, *args, **kwargs):
        nonlocal sequence
        sequence += 1
        start = time.perf_counter()
        event('conditional_gdn_begin', layer=indices[id(module)])
        def observer(head, begin, end, calls):
            event('conditional_gdn_tile', layer=indices[id(module)], head=head,
                  start=begin, stop=end, original_readout_calls=calls,
                  elapsed_seconds=time.perf_counter()-start)
        try:
            value = context.conditional_gdn_context(module, *args, observer=observer, **kwargs)
        except BaseException:
            event('conditional_gdn_failed', layer=indices[id(module)],
                  elapsed_seconds=time.perf_counter()-start)
            raise
        event('conditional_gdn_end', layer=indices[id(module)],
              elapsed_seconds=time.perf_counter()-start)
        return value

    runner.gdn_finite_pullback = partial(gdn.gdn_finite_pullback,
                                       conditional_context_pullback=callback)
    candidate = copy.copy(original)
    candidate.attribute = MethodType(runner.Qwen35DenseFiniteRunner.attribute, candidate)
    assert candidate.model is original.model and candidate.answer is original.answer
    assert candidate.capture_backend is original.capture_backend
    assert candidate.finite_fa is original.finite_fa and candidate.finite_fla is original.finite_fla
    assert original.attribute.__func__.__globals__['gdn_finite_pullback'] is native_gdn
    return candidate, dict(kind='conditional_gdn', runner_path=runner_path,
        runner_sha256=sha(runner_path), files=spec['files'],
        same_model_object=True, same_answer_owner=True, same_capture_backend=True,
        same_native_forward=True, same_FA=True, default_FLA_callback_unchanged=True,
        new_semantics='All-source factual conditional GDN memory, original head groups and native chunk tiles.',
        extra_model_forwards=0, extra_DT_calls=0, production_modified=False,
        accepted_candidate=False, phase_log=str(stream.name))
