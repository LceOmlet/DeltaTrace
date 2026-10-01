"""CPU lifecycle tests of the actual pinned owner methods, not model numerics."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from owner_rollout_scope import native_rollout_scope
from patch_owner_rollout_scope import patch_collector, patch_worker


OWNER = Path(os.environ.get('SCOPE_OWNER_ROOT', str(Path(__file__).resolve().parents[2] /
         'research/temporary/rl_upstream_alignment_20260929/recipe-sources/verl-agent-20bd331')))


def owner_worker_source():
    return (OWNER / 'verl/workers/fsdp_workers.py').read_text(encoding='utf-8')


def owner_methods(source):
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_only_generation_scope_changes():
    source = owner_worker_source()
    patched = patch_worker(source)
    assert patch_worker(patched) == patched
    before, after = owner_methods(source), owner_methods(patched)
    assert after.keys() - before.keys() == {'begin_rollout_context', 'end_rollout_context'}
    assert {name for name in before if ast.dump(before[name]) != ast.dump(after[name])} == {'generate_sequences'}
    collector = (OWNER / 'agent_system/multi_turn_rollout/rollout_loop.py').read_text(encoding='utf-8')
    fixed = patch_collector(collector)
    assert patch_collector(fixed) == fixed
    # The complete official environment loop body is unchanged.
    def loop(s):
        return next(n for n in ast.walk(ast.parse(s))
                    if isinstance(n, ast.FunctionDef) and n.name == 'multi_turn_loop')
    assert [ast.dump(n) for n in loop(collector).body] == [ast.dump(n) for n in loop(fixed).body]


class Data:
    def __init__(self):
        self.meta_info = {}

    def to(self, _):
        return self


def worker_fixture():
    calls = []

    class Manager:
        def __enter__(self):
            calls.append('enter')

        def __exit__(self, *exc):
            calls.append('exit')

        def preprocess_data(self, data):
            calls.append(('preprocess', data))
            return data

        def postprocess_data(self, data):
            calls.append(('postprocess', data))
            return data

    # Compile only the owner's actual three methods, without importing GPU
    # packages. Test doubles carry data/context events and do no computation.
    namespace = dict(register=lambda **_: lambda fn: fn,
                     Dispatch=SimpleNamespace(DP_COMPUTE_PROTO=1, ONE_TO_ALL=2),
                     DataProto=Data, logger=None, log_gpu_memory_usage=lambda *a, **k: None,
                     get_torch_device=lambda: SimpleNamespace(current_device=lambda: 'cpu', empty_cache=lambda: None))
    methods = owner_methods(patch_worker(owner_worker_source()))
    for name in ('begin_rollout_context', 'end_rollout_context', 'generate_sequences'):
        exec(compile(ast.Module(body=[methods[name]], type_ignores=[]), '<pinned owner method>', 'exec'), namespace)
    cls = type('OwnerMethods', (), {name: namespace[name] for name in (
        'begin_rollout_context', 'end_rollout_context', 'generate_sequences')})
    worker = cls()
    worker._is_rollout = True
    worker.generation_config = None
    worker.tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0)
    worker.rollout_sharding_manager = Manager()
    worker.rollout = SimpleNamespace(generate_sequences=lambda prompts: prompts)
    return worker, calls


def test_default_generation_keeps_original_context_and_data():
    worker, calls = worker_fixture()
    data = Data()
    assert worker.generate_sequences(data) is data
    assert calls == ['enter', ('preprocess', data), ('postprocess', data), 'exit']


def test_complete_rollout_enters_once_and_retains_original_per_call_transforms():
    worker, calls = worker_fixture()
    rows = [Data(), Data()]

    @native_rollout_scope
    def collect(self, data, worker_group, envs, is_train=True):
        assert envs == 'original environment' and is_train
        return [worker_group.generate_sequences(row) for row in data]

    result = collect(None, rows, worker, 'original environment')
    assert all(a is b for a, b in zip(result, rows))
    assert calls == ['enter', ('preprocess', rows[0]), ('postprocess', rows[0]),
                     ('preprocess', rows[1]), ('postprocess', rows[1]), 'exit']
    assert not worker._owner_rollout_context_open
    # A new rollout must resynchronize the current policy through the owner.
    collect(None, rows, worker, 'original environment')
    assert calls.count('enter') == calls.count('exit') == 2


def test_environment_failure_closes_owner_context_and_preserves_exception():
    worker, calls = worker_fixture()
    error = ValueError('environment error')

    @native_rollout_scope
    def collect(self, data, worker_group):
        worker_group.generate_sequences(data)
        raise error

    with pytest.raises(ValueError) as caught:
        collect(None, Data(), worker)
    assert caught.value is error
    assert calls[0] == 'enter' and calls[-1] == 'exit'
    assert not worker._owner_rollout_context_open
