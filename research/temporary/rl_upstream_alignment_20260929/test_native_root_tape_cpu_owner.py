"""CPU source/interface contracts for the isolated CPU root-tape owner seam.

Compare actual generated owners, not a remembered finite implementation. These
tests do not import Torch, run a model, or establish GPU numerical acceptance.
"""
from __future__ import annotations

import ast
import copy
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[2]
OWNER = Path(os.environ.get('DT_ROOT_TAPE_OWNER_SOURCE',
    REPO / 'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'))
OWNER_ROOT = Path(os.environ.get('DT_ROOT_INVENTORY_DECODER_SOURCE',
    REPO / 'deltatrace/clean/qwen35/qwen35_decoder_finite.py')).parent
ACCELERATED_ROOT = OWNER_ROOT.parent.parent / 'accelerated/qwen35'
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))

from prepare_native_root_tape_owner_20261004 import patched_owner
from prepare_native_root_tape_cpu_owner_20261005 import patched_cpu_root_owner, prepare
from prepare_native_root_capture_transport_owner_20261005 import (
    ACCELERATED_MODULES, MODULES, default_projection, patched_sources)
from test_native_root_tape_owner import (calls_named, method, normalized,
    owner_class, reconstructed_default_body)


class DisabledTransport(ast.NodeTransformer):
    """Project only the documented disabled root/transport controls."""
    def __init__(self, *, root_capture=False, no_tape=False):
        self.root_capture = root_capture
        self.no_tape = no_tape

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id == 'root_capture':
                return ast.copy_location(ast.Constant(value=self.root_capture), node)
            if node.id in ('decoder_transport', 'mixer_transport'):
                return ast.copy_location(ast.Dict(keys=[], values=[]), node)
            if self.no_tape and node.id == '_root_tape':
                return ast.copy_location(ast.Constant(value=None), node)
        return node

    def visit_Assign(self, node):
        names = {item.id for target in node.targets for item in ast.walk(target)
                 if isinstance(item, ast.Name)}
        if names & {'root_capture', 'decoder_transport', 'mixer_transport'}:
            return None
        return self.generic_visit(node)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if (len(node.ops) == len(node.comparators) == 1 and
                isinstance(node.left, ast.Constant) and
                isinstance(node.comparators[0], ast.Constant)):
            if isinstance(node.ops[0], ast.Is):
                return ast.copy_location(ast.Constant(
                    value=node.left.value is node.comparators[0].value), node)
            if isinstance(node.ops[0], ast.IsNot):
                return ast.copy_location(ast.Constant(
                    value=node.left.value is not node.comparators[0].value), node)
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        is_and = isinstance(node.op, ast.And)
        values = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                if (not bool(value.value) if is_and else bool(value.value)):
                    return value
                continue
            values.append(value)
        if not values:
            return ast.copy_location(ast.Constant(value=is_and), node)
        if len(values) == 1:
            return values[0]
        node.values = values
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [kw for kw in node.keywords if not (
            kw.arg is None and isinstance(kw.value, ast.Dict) and not kw.value.keys)]
        return node


def default_factory(cls):
    result = copy.deepcopy(method(cls, '_make_layer_captures'))
    if any(arg.arg == 'root_capture' for arg in result.args.kwonlyargs):
        position = next(i for i, arg in enumerate(result.args.kwonlyargs)
                        if arg.arg == 'root_capture')
        del result.args.kwonlyargs[position]
        del result.args.kw_defaults[position]
    return DisabledTransport().visit(result)


def default_body(cls):
    # The earlier scheduling normalizer verifies/project its known seam;
    # substitute this candidate's actual disabled capture factory first.
    projected = copy.deepcopy(cls)
    projected.body = [default_factory(cls) if isinstance(node, ast.FunctionDef)
                      and node.name == '_make_layer_captures' else node
                      for node in projected.body]
    body = reconstructed_default_body(projected)
    return DisabledTransport(no_tape=True).visit(body)


def actual_method_function(node, namespace=None):
    namespace = {} if namespace is None else namespace
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(module, '<actual-cpu-root-owner-method>', 'exec'), namespace)
    return namespace[node.name]


def actual_constructor_signature(source, class_name):
    """Inspect the real generated owner's constructor; never execute its body."""
    parsed = ast.parse(source)
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    return inspect.signature(actual_method_function(method(cls, '__init__')))


def bound_capture_classes(records, sources):
    """Record factory calls only after actual owner signatures accept them.

    This checks argument compatibility; it is not a substitute capture body.
    The derived accelerated import/MRO binding is checked independently below.
    """
    classes = {}
    for module, name in (
            ('native_root_decoder_transport_candidate.py', 'NativeDecoderCapture'),
            ('native_root_dense_attention_transport_candidate.py', 'NativeDenseAttentionCapture'),
            ('native_root_gdn_transport_candidate.py', 'NativeGDNCapture')):
        signature = actual_constructor_signature(sources[module], name)
        def initializer(self, *args, _signature=signature, **kwargs):
            _signature.bind(self, *args, **kwargs)
            records.append((args, kwargs))
        classes[name] = type(name, (), {'__init__': initializer,
            'actual_owner_signature': signature, 'actual_owner_source': module})
    return SimpleNamespace(**classes)


class RootTapeCPUOwnerSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = OWNER.read_bytes()
        cls.gpu_bytes, cls.gpu_metadata = patched_owner(cls.original)
        cls.cpu_bytes, cls.cpu_metadata = patched_cpu_root_owner(cls.original)
        cls.gpu_tree, cls.cpu_tree = [ast.parse(value.decode('utf8'))
                                    for value in (cls.gpu_bytes, cls.cpu_bytes)]
        cls.gpu, cls.cpu = [owner_class(value) for value in (cls.gpu_tree, cls.cpu_tree)]
        cls.transport_sources, cls.transport_metadata = patched_sources({
            name: (OWNER_ROOT / name).read_bytes()
            for name in ('native_attention_capture.py', 'native_dense_attention_capture.py',
                         'qwen35_decoder_finite.py', 'qwen35_gdn_finite.py')})

    def test_real_six_source_prepare_entry_creates_complete_isolated_artifacts(self):
        owner_root = OWNER_ROOT
        accelerated_root = ACCELERATED_ROOT
        originals = {name: ((owner_root if name in MODULES else accelerated_root) / name).read_bytes()
                     for name in MODULES | ACCELERATED_MODULES}
        paths = {name: (owner_root if name in MODULES else accelerated_root) / name
                 for name in originals}
        with tempfile.TemporaryDirectory(prefix='dt-cpu-root-entry-') as temporary:
            destination = Path(temporary) / 'isolated'
            self.assertFalse(destination.exists())
            metadata = prepare(OWNER, owner_root, destination,
                               hashlib.sha256(self.original).hexdigest())
            expected = set(MODULES.values()) | set(ACCELERATED_MODULES.values()) | {
                'qwen35_dense_finite_runner_root_tape_candidate.py', 'native_qwen35_root_tape.py'}
            self.assertEqual({path.name for path in destination.glob('*.py')}, expected)
            self.assertTrue((destination / 'prepared.json').is_file())
            self.assertTrue((destination / 'prepared-cpu-root.json').is_file())
            self.assertFalse(metadata['formal_deployment'])
            self.assertFalse(metadata['default_enabled'])
            transport = metadata['transport']
            self.assertEqual(set(transport['files']), set(originals))
            self.assertTrue(transport['accelerated_import_bindings_prepared'])
            self.assertFalse(transport['accelerated_runtime_backend_verified'])
            self.assertFalse(transport['finite_math_changed'])
            self.assertFalse(transport['parameter_prepare_release_changed'])
            self.assertFalse(transport['new_offloader_stream_or_cache'])
            self.assertEqual(transport['new_defaults'], {
                'preserve_strides': False, 'pinned_host': False,
                'defer_host_sync': False, 'restore_captures': False})
            for name, original in originals.items():
                manifest = transport['files'][name]
                self.assertEqual(paths[name].read_bytes(), original)
                self.assertEqual(manifest['owner_sha256'], hashlib.sha256(original).hexdigest())
                candidate_path = destination / manifest['candidate_filename']
                candidate = candidate_path.read_bytes()
                self.assertEqual(manifest['candidate_sha256'], hashlib.sha256(candidate).hexdigest())
                self.assertTrue(manifest['source_contracts']['default_projection_equals_original_module_ast'])
                self.assertEqual(normalized(default_projection(ast.parse(candidate), name)),
                                 normalized(ast.parse(original)), name)
            self.assertEqual(OWNER.read_bytes(), self.original)
            self.assertEqual(json.loads((destination / 'prepared-cpu-root.json').read_text()), metadata)
            self.assertEqual((destination / 'qwen35_dense_finite_runner_root_tape_candidate.py').read_bytes(),
                             self.cpu_bytes)
            # Reusing the actual preparation entry must not partially rewrite
            # the candidate or receipts before raising for an existing target.
            before = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}
            with self.assertRaises(FileExistsError):
                prepare(OWNER, owner_root, destination, hashlib.sha256(self.original).hexdigest())
            after = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}
            self.assertEqual(after, before)

    def test_real_prepare_rejects_wrong_locked_runner_before_creating_directory(self):
        with tempfile.TemporaryDirectory(prefix='dt-cpu-root-sha-') as temporary:
            destination = Path(temporary) / 'isolated'
            with self.assertRaisesRegex(ValueError, 'verified runner SHA'):
                prepare(OWNER, OWNER_ROOT, destination, '0' * 64)
            self.assertFalse(destination.exists())

    def test_generation_keeps_original_and_binds_exact_gpu_parent(self):
        self.assertEqual(OWNER.read_bytes(), self.original)
        self.assertEqual(self.cpu_metadata['owner_source_sha256'],
                         hashlib.sha256(self.original).hexdigest())
        self.assertEqual(self.cpu_metadata['parent_gpu_tape_source_sha256'],
                         hashlib.sha256(self.gpu_bytes).hexdigest())
        self.assertEqual(self.cpu_metadata['candidate_source_sha256'],
                         hashlib.sha256(self.cpu_bytes).hexdigest())
        self.assertFalse(self.cpu_metadata['formal_deployment'])
        self.assertFalse(self.cpu_metadata['default_enabled'])

    def test_disabled_capture_factory_matches_actual_gpu_owner_ast(self):
        self.assertEqual(normalized(default_factory(self.cpu)),
                         normalized(method(self.gpu, '_make_layer_captures')))

    def test_complete_disabled_body_matches_actual_gpu_owner_ast(self):
        self.assertEqual(normalized(default_body(self.cpu)),
                         normalized(default_body(self.gpu)))

    def test_other_methods_and_module_numeric_helpers_are_unchanged(self):
        for old in self.gpu.body:
            if isinstance(old, ast.FunctionDef) and old.name not in (
                    '_make_layer_captures', '_attribute_owner_body'):
                self.assertEqual(normalized(method(self.cpu, old.name)), normalized(old), old.name)
        for old in self.gpu_tree.body:
            if isinstance(old, ast.FunctionDef):
                new = next(item for item in self.cpu_tree.body
                           if isinstance(item, ast.FunctionDef) and item.name == old.name)
                self.assertEqual(normalized(new), normalized(old), old.name)

    def test_every_finite_call_changes_only_declared_restore_transport(self):
        candidate = copy.deepcopy(method(self.cpu, '_attribute_owner_body'))
        changed = 0
        for call in ast.walk(candidate):
            if not isinstance(call, ast.Call):
                continue
            if isinstance(call.func, ast.Name) and call.func.id == 'decoder_finite_pullback':
                extra = [kw for kw in call.keywords if kw.arg is None]
                self.assertEqual(len(extra), 1)
                expected = ast.parse("({'restore_captures':True} if _root_tape is not None "
                                     "and observer is None else {})", mode='eval').body
                self.assertEqual(normalized(extra[0].value), normalized(expected))
                call.keywords.remove(extra[0])
                changed += 1
        self.assertEqual(changed, 1)
        names = {'categorical_head_logits', 'selected_target_log_probs', 'answer',
                 'norm_residual', 'attention_finite_pullback', 'gdn_finite_pullback',
                 'decoder_finite_pullback', 'flash_attn_func'}
        self.assertCountEqual(calls_named(candidate, names), calls_named(
            method(self.gpu, '_attribute_owner_body'), names))

    def test_native_model_and_parameter_callback_calls_are_unchanged(self):
        old, new = [method(owner, '_attribute_owner_body') for owner in (self.gpu, self.cpu)]
        for name in ('model', 'prepare_layer', 'release_layer', 'replay_finite_layer'):
            self.assertEqual(calls_named(new, {name}), calls_named(old, {name}), name)

    def test_offload_field_options_and_global_batch_are_not_redefined(self):
        gpu = method(self.gpu, '_make_layer_captures')
        cpu = default_factory(self.cpu)
        names = {'gdn_gpu_capture_names', 'gdn_head_batch_size', 'compact_gdn_captures',
                 'copy_replay_captures', 'capture_backend'}
        for name in names:
            selected = lambda node: [normalized(value) for value in ast.walk(node)
                if isinstance(value, ast.Attribute) and value.attr == name]
            self.assertEqual(selected(cpu), selected(gpu), name)

    def factory(self, owner, backend=None, observer=None, is_fa=True, **controls):
        records = []
        captures = bound_capture_classes(records, self.transport_sources)
        symbols = dict(NativeDecoderCapture=captures.NativeDecoderCapture,
            NativeDenseAttentionCapture=captures.NativeDenseAttentionCapture,
            NativeGDNCapture=captures.NativeGDNCapture, flash_attention_forward=object(),
            flash_attn_varlen_func=object(), flash_attn_func=object(),
            _root_capture_backend=captures)
        if backend == 'ordinary_private':
            backend = captures
        if backend == 'diagnostic':
            # Equal exported classes do not make this the exact private
            # ordinary module. Identity must keep this diagnostic synchronous.
            backend = SimpleNamespace(**vars(captures))
        view = SimpleNamespace(copy_replay_captures=False, offload_replay_mixer=False,
            capture_backend=backend, pin_root_host=True, pin_replay_host=True,
            compact_gdn_captures=True, gdn_gpu_capture_names=('q', 'k'))
        layer = SimpleNamespace(self_attn=object(), linear_attn=object())
        fn = actual_method_function(method(owner, '_make_layer_captures'), symbols)
        result = fn(view, layer, is_fa, observer, 64, **controls)
        return records, result

    def test_default_factory_actual_cpu_spies_keep_original_arguments(self):
        # Compare argument identities in a single namespace, not separate
        # fabricated forward/model outputs.
        records = []
        captures = bound_capture_classes(records, self.transport_sources)
        namespace = dict(NativeDecoderCapture=captures.NativeDecoderCapture,
            NativeDenseAttentionCapture=captures.NativeDenseAttentionCapture,
            NativeGDNCapture=captures.NativeGDNCapture, flash_attention_forward=object(),
            flash_attn_varlen_func=object(), flash_attn_func=object())
        view = SimpleNamespace(copy_replay_captures=False, offload_replay_mixer=False,
            capture_backend=None, pin_root_host=True, pin_replay_host=True,
            compact_gdn_captures=True, gdn_gpu_capture_names=('q', 'k'))
        layer = SimpleNamespace(self_attn=object(), linear_attn=object())
        for is_fa in (True, False):
            for observer in (None, object()):
                records.clear()
                old = actual_method_function(method(self.gpu, '_make_layer_captures'), namespace)
                old(view, layer, is_fa, observer, 64)
                expected = list(records)
                records.clear()
                new = actual_method_function(method(self.cpu, '_make_layer_captures'), namespace)
                new(view, layer, is_fa, observer, 64, root_capture=False)
                self.assertEqual(records, expected)

    def test_cpu_root_uses_owner_transport_and_diagnostic_backend_does_not_defer(self):
        records, result = self.factory(self.cpu, root_capture=True)
        self.assertEqual(records[0][1]['destination'], 'cpu')
        self.assertTrue(records[0][1]['preserve_strides'])
        self.assertTrue(records[0][1]['pinned_host'])
        self.assertTrue(records[0][1]['defer_host_sync'])
        self.assertEqual(records[1][1]['destination'], 'cpu')
        self.assertTrue(records[1][1]['defer_host_sync'])
        self.assertTrue(result[2])
        records, _ = self.factory(self.cpu, backend='diagnostic', root_capture=True)
        self.assertFalse(records[0][1]['defer_host_sync'])
        self.assertFalse(records[1][1]['defer_host_sync'])

    def test_actual_generated_signatures_reject_the_prior_misspelled_keyword(self):
        for module, name, required in (
                ('native_root_decoder_transport_candidate.py', 'NativeDecoderCapture', 1),
                ('native_root_dense_attention_transport_candidate.py', 'NativeDenseAttentionCapture', 4),
                ('native_root_gdn_transport_candidate.py', 'NativeGDNCapture', 1)):
            signature = actual_constructor_signature(self.transport_sources[module], name)
            self.assertIn('defer_host_sync', signature.parameters)
            self.assertNotIn('defer_sync', signature.parameters)
            with self.assertRaisesRegex(TypeError, 'defer_sync'):
                signature.bind(None, *[object() for _ in range(required)], defer_sync=True)

    def test_private_backend_and_diagnostic_identity_cover_fa_and_gdn_signatures(self):
        for is_fa in (True, False):
            for backend, expected_defer in ((None, True), ('ordinary_private', True),
                                            ('diagnostic', False)):
                records, offload = self.factory(self.cpu, backend=backend, is_fa=is_fa,
                                                root_capture=True)
                self.assertTrue(offload[2])
                self.assertEqual(len(records), 2)
                for _, kwargs in records:
                    self.assertEqual(kwargs['defer_host_sync'], expected_defer)
                self.assertEqual(records[0][1]['destination'], 'cpu')
                self.assertEqual(records[1][1]['destination' if is_fa else 'device'], 'cpu')

    def test_root_ordinary_backend_import_and_defer_condition_are_exact(self):
        imports = [node for node in self.cpu_tree.body if isinstance(node, ast.Import)]
        selected = [alias for node in imports for alias in node.names
                    if alias.asname == '_root_capture_backend']
        self.assertEqual([(node.name, node.asname) for node in selected], [
            ('native_root_code_local_transport_candidate', '_root_capture_backend')])
        factory = method(self.cpu, '_make_layer_captures')
        conditions = [value for node in ast.walk(factory) if isinstance(node, ast.Dict)
                      for key, value in zip(node.keys, node.values)
                      if isinstance(key, ast.Constant) and key.value == 'defer_host_sync']
        expected = ast.parse('self.pin_root_host and (backend is None or '
                             'backend is _root_capture_backend)', mode='eval').body
        self.assertEqual(len(conditions), 2)
        self.assertTrue(all(normalized(value) == normalized(expected) for value in conditions))

    def test_observer_keeps_original_gpu_capture_and_no_deferred_transport(self):
        records, _ = self.factory(self.cpu, observer=object(), root_capture=True)
        self.assertEqual(records[0][1]['destination'], 'cuda')
        self.assertEqual(records[1][1]['destination'], 'cuda')
        self.assertNotIn('defer_host_sync', records[0][1])
        self.assertNotIn('defer_host_sync', records[1][1])

    def test_original_root_finally_fence_covers_exception_before_cleanup(self):
        body = method(self.cpu, '_attribute_owner_body')
        root_try = next(node for node in body.body if isinstance(node, ast.Try)
            and any(isinstance(item, ast.Constant)
                    and item.value == 'native_root_with_CPU_checkpoints' for item in ast.walk(node)))
        self.assertIsInstance(root_try.finalbody[0], ast.If)
        expected = ast.parse('self.pin_root_host or _root_tape is not None', mode='eval').body
        self.assertEqual(normalized(root_try.finalbody[0].test), normalized(expected))
        events = []
        def fail(**kwargs):
            events.append('original_root')
            raise ValueError('original root failure')
        torch = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append('sync'))))
        namespace = dict(torch=torch, root_context=nullcontext(),
            timed=lambda label, fn: fn(), paired_ids=object(), mask=object(),
            prefix_start=0, native_cache=None, selector=None,
            handles=[SimpleNamespace(remove=lambda: events.append('remove'))])
        arguments = ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'),
            ast.arg(arg='_root_tape'), ast.arg(arg='model')], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        function = ast.FunctionDef(name='invoke_original_root', args=arguments,
            body=[copy.deepcopy(root_try)], decorator_list=[])
        run = actual_method_function(function, namespace)
        with self.assertRaisesRegex(ValueError, 'original root failure'):
            run(SimpleNamespace(pin_root_host=False), object(), fail)
        self.assertEqual(events, ['original_root', 'sync', 'remove'])
        events.clear()
        with self.assertRaisesRegex(ValueError, 'original root failure'):
            run(SimpleNamespace(pin_root_host=False), None, fail)
        self.assertEqual(events, ['original_root', 'remove'])


if __name__ == '__main__':
    unittest.main()
