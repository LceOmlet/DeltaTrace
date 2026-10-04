"""Narrow root-tape owner contracts; no DT/GDN/FA numerical acceptance.

The standard-library cases compare actual owner AST and source artifacts.
The optional Torch cases use the existing real CPU decoder-hook fixture and
the exact NativeDecoderCapture class, without importing GPU finite kernels.
No model/rollout is executed by the source-only tests.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[2]
OWNER = Path(os.environ.get(
    'DT_ROOT_TAPE_OWNER_SOURCE',
    REPO / 'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'))
DECODER = Path(os.environ.get(
    'DT_ROOT_INVENTORY_DECODER_SOURCE',
    REPO / 'deltatrace/clean/qwen35/qwen35_decoder_finite.py'))
PREPARER = AUDIT / 'prepare_native_root_tape_owner_20261004.py'
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f'Cannot load diagnostic owner boundary: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tree(path: Path):
    return ast.parse(path.read_text(encoding='utf8'), filename=str(path))


def owner_class(parsed):
    return next(n for n in parsed.body if isinstance(n, ast.ClassDef)
                and n.name == 'Qwen35DenseFiniteRunner')


def method(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == name)


def normalized(node):
    return ast.dump(node, include_attributes=False)


def call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def calls_named(node, names):
    return [normalized(n) for n in ast.walk(node)
            if isinstance(n, ast.Call) and call_name(n) in names]


def original_capture_statements(attribute):
    reverse = next(n for n in attribute.body if isinstance(n, ast.For)
                   and isinstance(n.iter, ast.Call)
                   and call_name(n.iter) == 'reversed')
    start = next(i for i, n in enumerate(reverse.body)
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'copy_captures'
                         for t in n.targets))
    stop = next(i for i, n in enumerate(reverse.body[start:], start)
                if isinstance(n, ast.FunctionDef) and n.name == 'replay')
    return reverse.body[start:stop]


def public_wrapper_function(candidate):
    node = copy.deepcopy(method(candidate, 'attribute'))
    namespace = {}
    # Compile the exact public seam alone. The body spy below is only a
    # transport/lifecycle observation, not an alternate DT implementation.
    future = ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, '<actual-root-tape-public-wrapper>', 'exec'), namespace)
    return namespace['attribute'], namespace


def actual_cpu_fixture(torch):
    source = AUDIT / 'test_native_root_inventory_decoder_owner.py'
    required = {'original_decoder_capture', 'Decoder', 'MLP', 'NoMixerObservation'}
    nodes = [n for n in tree(source).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in required]
    if {n.name for n in nodes} != required:
        raise AssertionError('Existing original decoder-hook fixture interface changed')
    # Reuse its exact class/function AST. Import-only inventory/pytest modules
    # are not required for these fixtures and must not pull in another owner.
    namespace = {'torch': torch, 'ast': ast, 'os': os, 'Path': Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in required})


def reconstructed_default_body(candidate):
    """Project only the declared default-inert scheduling seam out of its AST.

    Capture construction is inlined from the actual candidate method, not
    substituted from a remembered implementation. Its separate test matches
    that method to the owner before this complete default-body comparison.
    """
    node = copy.deepcopy(method(candidate, '_attribute_owner_body'))
    capture_body = [copy.deepcopy(n) for n in method(candidate, '_make_layer_captures').body
                    if not isinstance(n, ast.Return)]

    class DefaultBranch(ast.NodeTransformer):
        def visit_If(self, item):
            test = item.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == '_root_tape' and len(test.ops) == 1
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                    and isinstance(test.ops[0], (ast.Is, ast.IsNot))):
                selected = item.body if isinstance(test.ops[0], ast.Is) else item.orelse
                statements = []
                for statement in selected:
                    transformed = self.visit(statement)
                    if isinstance(transformed, list):
                        statements.extend(transformed)
                    elif transformed is not None:
                        statements.append(transformed)
                return statements
            return self.generic_visit(item)

        def visit_Assign(self, item):
            names = {n.id for target in item.targets for n in ast.walk(target)
                     if isinstance(n, ast.Name)}
            if names == {'root_context'}:
                return None
            if (isinstance(item.value, ast.Call)
                    and call_name(item.value) == '_make_layer_captures'):
                return copy.deepcopy(capture_body)
            return self.generic_visit(item)

        def visit_With(self, item):
            item.items = [value for value in item.items
                          if not (isinstance(value.context_expr, ast.Name)
                                  and value.context_expr.id == 'root_context')]
            return self.generic_visit(item)

    node = DefaultBranch().visit(node)
    node.name = 'attribute'
    index = next(i for i, arg in enumerate(node.args.kwonlyargs)
                 if arg.arg == '_root_tape')
    del node.args.kwonlyargs[index]
    del node.args.kw_defaults[index]
    # These unchanged calculations were moved earlier solely so root captures
    # can use the same coefficient range. Restore their original location.
    moved = []
    remaining = []
    for item in node.body:
        names = {n.id for target in getattr(item, 'targets', []) for n in ast.walk(target)
                 if isinstance(n, ast.Name)}
        (moved if names & {'local_starts', 'gdn_cut'} else remaining).append(item)
    layout = next(i for i, item in enumerate(remaining)
                  if isinstance(item, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'layout' for t in item.targets))
    node.body = remaining[:layout + 1] + moved + remaining[layout + 1:]
    return node


class RootTapeSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PREPARER.exists():
            raise unittest.SkipTest('Root-tape preparer is not prepared yet')
        cls.owner_bytes = OWNER.read_bytes()
        cls.original_tree = tree(OWNER)
        cls.original = owner_class(cls.original_tree)
        cls.temporary = tempfile.TemporaryDirectory(prefix='dt-root-tape-cpu-')
        cls.directory = Path(cls.temporary.name) / 'candidate'
        cls.preparer = load_module(PREPARER, '_dt_root_tape_preparer_contract')
        cls.prepared = cls.preparer.prepare(
            OWNER, cls.directory, hashlib.sha256(cls.owner_bytes).hexdigest())
        cls.runner_path = next(p for p in cls.directory.glob('*.py')
                               if any(isinstance(n, ast.ClassDef)
                                      and n.name == 'Qwen35DenseFiniteRunner'
                                      for n in tree(p).body))
        cls.candidate_tree = tree(cls.runner_path)
        cls.candidate = owner_class(cls.candidate_tree)
        cls.helper_path = cls.directory / 'native_qwen35_root_tape.py'

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_preparer_does_not_change_owner_or_finite_sources(self):
        self.assertEqual(OWNER.read_bytes(), self.owner_bytes)
        self.assertTrue(self.helper_path.exists())
        # No finite/reward/trainer implementation is published by the seam.
        published = {p.name for p in self.directory.glob('*.py')}
        self.assertEqual(published, {self.runner_path.name, self.helper_path.name})

    def test_only_requested_constructor_flag_defaults_false(self):
        old = method(self.original, '__init__')
        new = copy.deepcopy(method(self.candidate, '__init__'))
        self.assertEqual([a.arg for a in new.args.kwonlyargs].count('reuse_root_captures'), 1)
        index = next(i for i, arg in enumerate(new.args.kwonlyargs)
                     if arg.arg == 'reuse_root_captures')
        self.assertEqual(normalized(new.args.kw_defaults[index]),
                         normalized(ast.Constant(value=False)))
        del new.args.kwonlyargs[index]
        del new.args.kw_defaults[index]
        new.body = [n for n in new.body if not (
            isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == 'reuse_root_captures'
                for t in n.targets))]
        self.assertEqual(normalized(new), normalized(old))

    def test_other_existing_methods_are_original_owner_ast(self):
        exceptions = {'__init__', 'attribute'}
        for old in self.original.body:
            if isinstance(old, ast.FunctionDef) and old.name not in exceptions:
                self.assertEqual(normalized(method(self.candidate, old.name)),
                                 normalized(old), old.name)

    def test_complete_default_body_reconstructs_the_exact_original_ast(self):
        self.assertEqual(normalized(reconstructed_default_body(self.candidate)),
                         normalized(method(self.original, 'attribute')))

    def test_module_level_numerical_helpers_are_original_owner_ast(self):
        for old in self.original_tree.body:
            if isinstance(old, ast.FunctionDef):
                new = next(n for n in self.candidate_tree.body
                           if isinstance(n, ast.FunctionDef) and n.name == old.name)
                self.assertEqual(normalized(new), normalized(old), old.name)

    def test_capture_factory_uses_the_original_construction_statements(self):
        old = original_capture_statements(method(self.original, 'attribute'))
        factory = method(self.candidate, '_make_layer_captures')
        body = [n for n in factory.body if not isinstance(n, ast.Return)]
        # A helper docstring is explanatory only; every executable statement
        # before the return must be the original owner's construction.
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        self.assertEqual([normalized(n) for n in body], [normalized(n) for n in old])

    def test_answer_and_finite_formula_calls_are_original_ast(self):
        names = {'categorical_head_logits', 'selected_target_log_probs', 'answer',
                 'norm_residual', 'attention_finite_pullback', 'gdn_finite_pullback',
                 'decoder_finite_pullback', 'flash_attn_func'}
        old = method(self.original, 'attribute')
        new = method(self.candidate, '_attribute_owner_body')
        self.assertCountEqual(calls_named(new, names), calls_named(old, names))

    def test_existing_seed_coefficient_and_finite_result_assignments_unchanged(self):
        targets = {'m', 'mnorm', 'seed', 'root_logp', 'root_lp0', 'root_lp1',
                   'lp0', 'lp1', 'effect_G', 'compiled_seed_G', 'seed_effect',
                   'new', 'terms', 'z', 'outcome_logits'}

        def assignments(node):
            found = []
            for item in ast.walk(node):
                if isinstance(item, ast.Assign):
                    names = {n.id for target in item.targets for n in ast.walk(target)
                             if isinstance(n, ast.Name)}
                    if names & targets:
                        found.append(normalized(item))
            return found

        self.assertCountEqual(assignments(method(self.candidate, '_attribute_owner_body')),
                              assignments(method(self.original, 'attribute')))

    def test_original_native_model_call_count_and_arguments_unchanged(self):
        old = method(self.original, 'attribute')
        new = method(self.candidate, '_attribute_owner_body')
        self.assertEqual(calls_named(new, {'model'}), calls_named(old, {'model'}))
        self.assertEqual(len(calls_named(new, {'model'})), 1)

    def test_public_signature_unchanged_and_default_dispatch_is_original_body(self):
        self.assertEqual(normalized(method(self.candidate, 'attribute').args),
                         normalized(method(self.original, 'attribute').args))
        wrapper, namespace = public_wrapper_function(self.candidate)
        tape_constructions, forwards = [], []

        class RecordingTape:
            def __init__(self, *args, **kwargs):
                tape_constructions.append((args, kwargs))

        namespace['NativeQwen35RootTape'] = RecordingTape
        result = object()

        def original_body(*args, **kwargs):
            forwards.append((args, kwargs))
            return result

        view = SimpleNamespace(reuse_root_captures=False,
                               _attribute_owner_body=original_body)
        endpoints, mask, selection, observer, provider = [object() for _ in range(5)]
        actual = wrapper(view, endpoints, mask, selection, False, observer,
                         prefix_cache_provider=provider)
        self.assertIs(actual, result)
        self.assertFalse(tape_constructions)
        self.assertEqual(len(forwards), 1)
        args, kwargs = forwards[0]
        self.assertIsNone(kwargs.pop('_root_tape', None))
        transported = inspect.signature(wrapper).bind(view, *args, **kwargs)
        expected = inspect.signature(wrapper).bind(
            view, endpoints, mask, selection, False, observer,
            prefix_cache_provider=provider)
        transported.apply_defaults()
        expected.apply_defaults()
        self.assertEqual(transported.arguments, expected.arguments)

    def test_enabled_dispatch_clears_tape_and_preserves_owner_exception(self):
        wrapper, namespace = public_wrapper_function(self.candidate)
        instances, forwards = [], []

        class RecordingTape:
            def __init__(self, *args, **kwargs):
                self.clear_count = 0
                instances.append(self)

            def clear(self):
                self.clear_count += 1

        namespace['NativeQwen35RootTape'] = RecordingTape

        def original_body(*args, **kwargs):
            forwards.append(kwargs['_root_tape'])
            raise ValueError('original owner failure')

        view = SimpleNamespace(reuse_root_captures=True,
                               _attribute_owner_body=original_body)
        with self.assertRaisesRegex(ValueError, 'original owner failure'):
            wrapper(view, object(), object(), object())
        self.assertEqual(len(instances), 1)
        self.assertEqual(forwards, instances)
        self.assertEqual(instances[0].clear_count, 1)

    def test_original_parameter_prepare_and_release_still_called(self):
        old = method(self.original, 'attribute')
        new = method(self.candidate, '_attribute_owner_body')
        for name in ('prepare_layer', 'release_layer'):
            self.assertEqual(calls_named(new, {name}), calls_named(old, {name}))

    def test_replay_diagnostic_expression_is_not_fabricated(self):
        old = method(self.original, 'attribute')
        new = method(self.candidate, '_attribute_owner_body')
        for key in ('replay_relative_L2', 'replay_output_effect'):
            def values(node):
                return [normalized(value) for item in ast.walk(node)
                        if isinstance(item, ast.Dict)
                        for field, value in zip(item.keys, item.values)
                        if isinstance(field, ast.Constant) and field.value == key
                        and not (isinstance(value, ast.Constant) and value.value is None)]
            self.assertEqual(values(new), values(old), key)
        # No added scalar literal is accepted as a replay numerical result.
        # This compares only the original two diagnostic field expressions;
        # the root-tape branch may mark replay unavailable rather than emit it.

    def test_helper_never_invokes_a_layer_or_model_forward(self):
        helper = tree(self.helper_path)
        names = {call_name(n) for n in ast.walk(helper) if isinstance(n, ast.Call)}
        self.assertFalse(names & {'forward', 'forward_root', 'layer', 'model',
                                  'decoder_finite_pullback', 'gdn_finite_pullback',
                                  'attention_finite_pullback'})
        # Native hooks/capture contexts may call only the supplied factory,
        # their lifecycle APIs and metadata/cleanup; no alternative math.


class RootTapeRealDecoderCPUContracts(unittest.TestCase):
    """Original capture plus real Torch hooks only; no FA/GDN substitution."""
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest('Existing CPU Torch is unavailable') from error
        helper = AUDIT / 'native_qwen35_root_tape.py'
        if not helper.exists():
            raise unittest.SkipTest('Root-tape helper is not prepared yet')
        cls.torch = torch
        cls.fixture = actual_cpu_fixture(torch)
        cls.Tape = load_module(helper, '_actual_root_tape_helper_contract').NativeQwen35RootTape
        with mock.patch.dict(os.environ, {'DT_ROOT_INVENTORY_DECODER_SOURCE': str(DECODER)}):
            cls.Capture = cls.fixture.original_decoder_capture()

    def factory(self, created):
        def native_capture(index, layer):
            dc = self.Capture(layer, destination='cpu', copy_tensors=False,
                              retained_names={'input_norm_input', 'post_norm_input',
                                              'gate_output', 'up_output', 'silu_output'})
            mc = self.fixture.NoMixerObservation()
            created.append((index, dc, mc))
            return dc, mc, False
        return native_capture

    def test_original_decoder_artifacts_survive_root_and_equal_direct_capture(self):
        torch, layer = self.torch, self.fixture.Decoder().eval()
        value = torch.arange(96, dtype=torch.float32).reshape(8, 3, 4)
        with torch.no_grad(), self.Capture(
                layer, destination='cpu', copy_tensors=False,
                retained_names={'input_norm_input', 'post_norm_input', 'gate_output',
                                'up_output', 'silu_output'}) as reference:
            expected = layer(value)
        tape, created, native_calls = self.Tape(), [], []
        counter = layer.register_forward_hook(lambda *_args: native_calls.append(1))
        try:
            with torch.no_grad(), tape.capture_scope([layer], self.factory(created)):
                actual = layer(value)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(native_calls, [1])
            dc, mc, offload = tape.pop(0)
            self.assertFalse(offload)
            self.assertEqual(dc.calls, reference.calls)
            self.assertEqual(set(dc.values), set(reference.values))
            for name, operand in dc.values.items():
                self.assertTrue(torch.equal(operand, reference.values[name]), name)
                self.assertEqual(operand.dtype, reference.values[name].dtype)
                self.assertEqual(operand.stride(), reference.values[name].stride())
            report = tape.report()
            self.assertEqual(report['captured_layers'], 1)
            self.assertEqual(report['consumed_layers'], 1)
            self.assertEqual(report['unconsumed_layers'], 0)
            self.assertEqual(report['version_checks'], 5)
            self.assertEqual(report['snapshot_copies_added'], 0)
            self.assertFalse(dc.handles)
            dc.values.clear()
            mc.values.clear()
        finally:
            counter.remove()
            tape.clear()
            reference.values.clear()
        self.assertFalse(layer._forward_hooks or layer._forward_pre_hooks)

    def test_two_original_layers_are_captured_once_and_consumed_in_reverse_order(self):
        torch = self.torch
        layers = [self.fixture.Decoder().eval(), self.fixture.Decoder().eval()]
        tape, created = self.Tape(), []
        value = torch.zeros(8, 3, 4)
        try:
            with torch.no_grad(), tape.capture_scope(layers, self.factory(created)):
                for layer in layers:
                    value = layer(value)
            self.assertEqual([index for index, _dc, _mc in created], [0, 1])
            for index in (1, 0):
                dc, mc, _offload = tape.pop(index)
                self.assertEqual(dc.calls['decoder'], 1)
                dc.values.clear()
                mc.values.clear()
            self.assertEqual(tape.report()['consumed_layers'], 2)
            self.assertFalse(tape.entries or tape.versions or tape.handles)
        finally:
            tape.clear()
        self.assertTrue(all(not layer._forward_hooks and not layer._forward_pre_hooks
                            for layer in layers))

    def test_original_forward_failure_and_partial_capture_cleanup(self):
        torch, layer = self.torch, self.fixture.Decoder().eval()
        tape, created = self.Tape(), []
        try:
            with self.assertRaisesRegex(ValueError, 'original forward failure'):
                with torch.no_grad(), tape.capture_scope([layer], self.factory(created)):
                    layer(torch.zeros(8, 3, 4), fail=True)
        finally:
            tape.clear()
        self.assertFalse(layer._forward_hooks or layer._forward_pre_hooks)
        self.assertFalse(tape.entries or tape.versions or tape.handles or tape.active)
        self.assertTrue(all(not dc.values and not dc.handles and not mc.values
                            for _index, dc, mc in created))

    def test_actual_borrowed_tensor_version_check_rejects_mutation(self):
        torch, layer = self.torch, self.fixture.Decoder().eval()
        tape, created = self.Tape(), []
        value = torch.zeros(8, 3, 4)
        try:
            with torch.no_grad(), tape.capture_scope([layer], self.factory(created)):
                layer(value)
            value.add_(1)
            with self.assertRaisesRegex(RuntimeError, 'Original root capture mutated'):
                tape.pop(0)
            self.assertEqual(tape.report()['consumed_layers'], 0)
        finally:
            tape.clear()
        self.assertFalse(tape.entries or tape.versions or tape.handles)


if __name__ == '__main__':
    unittest.main()
