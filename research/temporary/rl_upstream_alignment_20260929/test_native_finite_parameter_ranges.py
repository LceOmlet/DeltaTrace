"""Owner-spy/source contracts for passive parameter-phase ranges.

The standard-library spy checks dispatch and cleanup, not Torch/FSDP behavior.
The optional CPU Torch test checks actual record_function event names only.
Neither test runs a model, parameter movement, GPU operation or collective.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
from unittest import mock

import pytest


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[2]
HELPER = AUDIT / 'native_finite_parameter_ranges.py'
OWNER = Path(os.environ.get('DT_FINITE_PARAMETER_RANGE_OWNER_SOURCE',
    REPO / 'experiments/rl/deltatrace_rollout.py'))


def load_helper():
    spec = importlib.util.spec_from_file_location('_passive_parameter_ranges', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NativeFiniteParameterRanges


class OwnerFailure(RuntimeError):
    pass


class OwnerSpy:
    def __init__(self):
        self.layers = [object(), object()]
        self.model = types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=self.layers))
        self.calls = []
        self.prepare_result, self.release_result = object(), object()
        self.failure = None

    def prepare_finite_layer(self, layer):
        self.calls.append(('prepare', layer))
        if self.failure is not None:
            raise self.failure
        return self.prepare_result

    def release_finite_layer(self, layer):
        self.calls.append(('release', layer))
        return self.release_result


class RangeSpy:
    def __init__(self):
        self.events = []

    def record_function(self, name):
        events = self.events
        class Range:
            def __enter__(self):
                events.append(('enter', name))

            def __exit__(self, exc_type, exc, traceback):
                events.append(('exit', name, exc))
                return False
        return Range()


def test_actual_callback_signature_and_no_parameter_owner_in_helper():
    owner = next(node for node in ast.parse(OWNER.read_bytes()).body
                 if isinstance(node, ast.ClassDef) and node.name == '_Qwen35CausalOwnerView')
    for name in ('prepare_finite_layer', 'release_finite_layer'):
        method = next(node for node in owner.body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
        assert [arg.arg for arg in method.args.args] == ['self', 'layer']
        assert not method.args.vararg and not method.args.kwarg and not method.args.kwonlyargs
    tree = ast.parse(HELPER.read_bytes())
    forbidden = {'unshard', 'reshard', 'set_modules_to_forward_prefetch',
                 'all_gather', 'all_reduce', 'cuda', 'to', 'copy', 'clone', 'synchronize'}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden]
    wrapped = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == 'labelled')
    returned = next(node for node in ast.walk(wrapped) if isinstance(node, ast.Return))
    assert ast.unparse(returned.value) == 'original(layer)'


@pytest.mark.parametrize('instance_methods', [False, True])
def test_same_bound_methods_args_returns_ranges_and_exact_restoration(instance_methods):
    owner, ranges = OwnerSpy(), RangeSpy()
    names = ('prepare_finite_layer', 'release_finite_layer')
    originals = {name: getattr(owner, name) for name in names}
    if instance_methods:
        for name in names:
            setattr(owner, name, originals[name])
    before = vars(owner).copy()
    with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(profiler=ranges)}):
        with load_helper()(owner):
            assert owner.prepare_finite_layer(owner.layers[1]) is owner.prepare_result
            assert owner.release_finite_layer(layer=owner.layers[1]) is owner.release_result
    assert vars(owner).keys() == before.keys()
    for name in names:
        assert getattr(owner, name) == originals[name]
        if instance_methods:
            assert vars(owner)[name] is before[name]
    assert owner.calls == [('prepare', owner.layers[1]), ('release', owner.layers[1])]
    assert ranges.events == [
        ('enter', 'DT_native_prepare_finite_layer_1'),
        ('exit', 'DT_native_prepare_finite_layer_1', None),
        ('enter', 'DT_native_release_finite_layer_1'),
        ('exit', 'DT_native_release_finite_layer_1', None)]


def test_original_exception_identity_restored_and_context_reusable():
    owner, ranges = OwnerSpy(), RangeSpy()
    failure = OwnerFailure('original owner failure')
    owner.failure = failure
    context = load_helper()(owner)
    with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(profiler=ranges)}):
        with pytest.raises(OwnerFailure) as observed:
            with context:
                owner.prepare_finite_layer(owner.layers[0])
        assert observed.value is failure
        assert not context.active and context.saved == []
        assert 'prepare_finite_layer' not in vars(owner)
        assert 'release_finite_layer' not in vars(owner)
        owner.failure = None
        with context:
            assert owner.prepare_finite_layer(owner.layers[0]) is owner.prepare_result
    assert ranges.events[1] == ('exit', 'DT_native_prepare_finite_layer_0', failure)


def test_body_exception_is_not_suppressed_and_callbacks_restore():
    owner, ranges = OwnerSpy(), RangeSpy()
    failure = OwnerFailure('attribute failure outside callbacks')
    with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(profiler=ranges)}):
        with pytest.raises(OwnerFailure) as observed:
            with load_helper()(owner):
                raise failure
    assert observed.value is failure and owner.calls == [] and ranges.events == []
    assert 'prepare_finite_layer' not in vars(owner) and 'release_finite_layer' not in vars(owner)


def test_actual_cpu_torch_profiler_records_only_existing_owner_calls():
    torch = pytest.importorskip('torch')
    owner = OwnerSpy()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with load_helper()(owner):
            owner.prepare_finite_layer(owner.layers[0])
            owner.release_finite_layer(owner.layers[0])
            owner.prepare_finite_layer(owner.layers[1])
            owner.release_finite_layer(owner.layers[1])
    keys = [event.key for event in profile.events() if event.key.startswith('DT_native_')]
    assert keys == ['DT_native_prepare_finite_layer_0', 'DT_native_release_finite_layer_0',
                    'DT_native_prepare_finite_layer_1', 'DT_native_release_finite_layer_1']
    assert owner.calls == [('prepare', owner.layers[0]), ('release', owner.layers[0]),
                           ('prepare', owner.layers[1]), ('release', owner.layers[1])]
    assert 'prepare_finite_layer' not in vars(owner) and 'release_finite_layer' not in vars(owner)
