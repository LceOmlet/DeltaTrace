"""Source and CPU transport contracts; no FA/FLA or DT numerical acceptance.

Protocol spies check only staging/cleanup order. Optional real CPU Torch tests
use the original decoder hook fixture and original copy_capture_tensor body.
No source-only case imports Torch or executes a GPU/model operation.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[2]
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))
import prepare_native_root_capture_transport_owner_20261005 as preparer


def ast_dump(node):
    return ast.dump(node, include_attributes=False)


def top_node(source, name):
    return next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)


def class_method(source, class_name, name):
    return next(n for n in top_node(source, class_name).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def execute_nodes(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=copy.deepcopy(nodes), type_ignores=[])), '<actual-owner-AST>', 'exec'), namespace)


def owner_roots():
    source = os.environ.get('DT_ROOT_INVENTORY_DECODER_SOURCE')
    owner_root = Path(source).parent if source else REPO / 'deltatrace/clean/qwen35'
    return owner_root, owner_root.parents[1] / 'accelerated/qwen35'


class TransportSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner_root, cls.accelerated_owner_root = owner_roots()
        cls.originals = {name: (cls.owner_root / name).read_bytes() for name in preparer.MODULES}
        cls.accelerated_originals = {name: (cls.accelerated_owner_root / name).read_bytes()
                                    for name in preparer.ACCELERATED_MODULES}
        cls.candidates, cls.metadata = preparer.patched_sources(cls.originals,
            accelerated_originals=cls.accelerated_originals)

    def candidate(self, original_name):
        return self.candidates[(preparer.MODULES | preparer.ACCELERATED_MODULES)[original_name]]

    def test_all_original_owner_sha_identities(self):
        for name, source in self.originals.items():
            self.assertEqual(preparer.sha256(source), preparer.EXPECTED_OWNER_SHA256[name], name)
        for name, source in self.accelerated_originals.items():
            self.assertEqual(preparer.sha256(source), preparer.EXPECTED_ACCELERATED_SHA256[name], name)

    def test_all_disabled_extensions_project_to_complete_original_ast(self):
        for name, original in (self.originals | self.accelerated_originals).items():
            projected = preparer.default_projection(ast.parse(self.candidate(name)), name)
            self.assertEqual(ast_dump(ast.parse(original)), ast_dump(projected), name)

    def test_original_copy_helper_body_is_identical(self):
        name = 'native_attention_capture.py'
        self.assertEqual(ast_dump(top_node(self.originals[name], 'copy_capture_tensor')),
                         ast_dump(top_node(self.candidate(name), 'copy_capture_tensor')))

    def test_original_finite_math_functions_remain_identical(self):
        for name in ('qwen35_decoder_finite.py', 'qwen35_gdn_finite.py'):
            original_functions = [n for n in ast.parse(self.originals[name]).body
                                  if isinstance(n, ast.FunctionDef)
                                  and n.name != 'decoder_finite_pullback']
            for original in original_functions:
                self.assertEqual(ast_dump(original), ast_dump(top_node(
                    self.candidate(name), original.name)), original.name)

    def test_original_decoder_retain_assignment_is_default_else(self):
        original = class_method(self.originals['qwen35_decoder_finite.py'],
                                'NativeDecoderCapture', 'retain')
        candidate = class_method(self.candidate('qwen35_decoder_finite.py'),
                                 'NativeDecoderCapture', 'retain')
        self.assertEqual(ast_dump(original.body[0].body[0]),
                         ast_dump(candidate.body[0].body[0].orelse[0]))

    def test_private_imports_do_not_replace_default_modules(self):
        dense_imports = [n.module for n in ast.parse(self.candidate(
            'native_dense_attention_capture.py')).body if isinstance(n, ast.ImportFrom)]
        self.assertIn('native_root_attention_transport_candidate', dense_imports)
        self.assertNotIn('native_attention_capture', dense_imports)
        gdn_imports = [n.module for n in ast.parse(self.candidate(
            'qwen35_gdn_finite.py')).body if isinstance(n, ast.ImportFrom)]
        self.assertIn('native_root_decoder_transport_candidate', gdn_imports)
        self.assertIn('native_root_attention_transport_candidate', gdn_imports)

    def test_accelerated_candidates_change_only_imports(self):
        for name, source in self.accelerated_originals.items():
            before = [n for n in ast.parse(source).body
                      if not isinstance(n, (ast.Import, ast.ImportFrom))]
            after = [n for n in ast.parse(self.candidate(name)).body
                     if not isinstance(n, (ast.Import, ast.ImportFrom))]
            self.assertEqual([ast_dump(n) for n in before], [ast_dump(n) for n in after], name)
            contract = self.metadata['files'][name]['source_contracts']
            self.assertIs(contract['all_non_import_ast_identical'], True)
            self.assertIs(contract['imports_only'], True)

    def test_accelerated_imports_bind_private_owners_and_original_events(self):
        retained = [n for n in ast.parse(self.candidate('qwen35_retained_capture.py')).body
                    if isinstance(n, ast.ImportFrom)]
        self.assertEqual({n.module for n in retained}, {
            'native_root_attention_transport_candidate', 'native_root_decoder_transport_candidate',
            'native_root_dense_attention_transport_candidate', 'native_root_gdn_transport_candidate'})
        self.assertTrue(all(n.level == 0 for n in retained))
        local = [n for n in ast.parse(self.candidate('qwen35_code_local_capture.py')).body
                 if isinstance(n, ast.ImportFrom)]
        self.assertEqual({n.module for n in local}, {
            'accelerated.native_capture_events', 'native_root_retained_transport_candidate'})
        self.assertTrue(all(n.level == 0 for n in local))
        self.assertEqual(next(n.names[0].name for n in local
            if n.module == 'accelerated.native_capture_events'), 'LocalCaptureEvents')
        self.assertEqual([n.names[0].name for n in local
            if n.module == 'native_root_retained_transport_candidate'],
            ['NativeDecoderCapture', 'NativeGDNCapture', 'NativeDenseAttentionCapture'])

    def test_recorded_owner_path_environment_takes_precedence(self):
        saved = os.environ.get('DT_ROOT_INVENTORY_DECODER_SOURCE')
        try:
            source = Path('recorded/deltatrace/clean/qwen35/qwen35_decoder_finite.py')
            os.environ['DT_ROOT_INVENTORY_DECODER_SOURCE'] = str(source)
            clean, accelerated = owner_roots()
            self.assertEqual(clean, source.parent)
            self.assertEqual(accelerated, source.parent.parents[1] / 'accelerated/qwen35')
        finally:
            if saved is None:
                os.environ.pop('DT_ROOT_INVENTORY_DECODER_SOURCE', None)
            else:
                os.environ['DT_ROOT_INVENTORY_DECODER_SOURCE'] = saved

    def test_new_constructor_and_finite_keywords_default_false(self):
        for name, class_name in (('native_attention_capture.py', 'NativeAttentionCapture'),
                ('native_dense_attention_capture.py', 'NativeDenseAttentionCapture'),
                ('qwen35_decoder_finite.py', 'NativeDecoderCapture'),
                ('qwen35_gdn_finite.py', 'NativeGDNCapture')):
            fn = class_method(self.candidate(name), class_name, '__init__')
            defaults = dict(zip((a.arg for a in fn.args.kwonlyargs), fn.args.kw_defaults))
            self.assertIs(defaults['defer_host_sync'].value, False)
        decoder = top_node(self.candidate('qwen35_decoder_finite.py'), 'decoder_finite_pullback')
        defaults = dict(zip((a.arg for a in decoder.args.kwonlyargs), decoder.args.kw_defaults))
        self.assertIs(defaults['restore_captures'].value, False)

    def test_preparer_writes_private_manifest_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'private-transport'
            manifest = preparer.prepare(self.owner_root, destination,
                accelerated_owner_root=self.accelerated_owner_root)
            self.assertEqual(set(self.candidates), {p.name for p in destination.glob('*.py')})
            self.assertTrue((destination / 'prepared.json').exists())
            self.assertEqual(len(list(destination.glob('*.patch'))), 6)
            json.dumps(manifest)
            for name, source in self.originals.items():
                self.assertEqual((self.owner_root / name).read_bytes(), source)
                self.assertTrue(manifest['files'][name]['source_contracts'][
                    'default_projection_equals_original_module_ast'])
            for name, source in self.accelerated_originals.items():
                self.assertEqual((self.accelerated_owner_root / name).read_bytes(), source)
                self.assertTrue(manifest['files'][name]['source_contracts'][
                    'all_non_import_ast_identical'])
            with self.assertRaises(FileExistsError):
                preparer.prepare(self.owner_root, destination,
                    accelerated_owner_root=self.accelerated_owner_root)

    def test_sha_mismatch_writes_nothing(self):
        expected = dict(preparer.EXPECTED_OWNER_SHA256)
        expected['qwen35_decoder_finite.py'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'private-transport'
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                preparer.prepare(self.owner_root, destination, expected,
                    accelerated_owner_root=self.accelerated_owner_root)
            self.assertFalse(destination.exists())

    def test_accelerated_sha_mismatch_writes_nothing(self):
        expected = dict(preparer.EXPECTED_ACCELERATED_SHA256)
        expected['qwen35_retained_capture.py'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'private-transport'
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                preparer.prepare(self.owner_root, destination,
                    accelerated_owner_root=self.accelerated_owner_root,
                    expected_accelerated_sha256=expected)
            self.assertFalse(destination.exists())

    def test_exit_fence_default_and_deferred_in_normal_and_exception_paths(self):
        # Execute the actual patched owner exit methods with metadata-only
        # stream/profile spies. This checks no Tensor/GPU numerical behavior.
        for name, class_name in (('native_attention_capture.py', 'NativeAttentionCapture'),
                                 ('qwen35_gdn_finite.py', 'NativeGDNCapture'),
                                 ('qwen35_decoder_finite.py', 'NativeDecoderCapture')):
            fn = class_method(self.candidate(name), class_name, '__exit__')
            for deferred in (False, True):
                for exception in (False, True):
                    syncs, profile_changes, removed = [], [], []
                    namespace = dict(torch=SimpleNamespace(
                        device=lambda _: SimpleNamespace(type='cpu'),
                        cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(
                            synchronize=lambda: syncs.append('original-current-stream')))),
                        sys=SimpleNamespace(setprofile=lambda value: profile_changes.append(value)))
                    execute_nodes([fn], namespace)
                    obj = SimpleNamespace(pinned_host=True, defer_host_sync=deferred,
                        device='cpu', destination='cpu', handles=[SimpleNamespace(
                            remove=lambda: removed.append('original-handle'))])
                    error = RuntimeError('actual-capture-failure') if exception else None
                    namespace['__exit__'](obj, type(error) if error else None, error, None)
                    self.assertEqual(len(syncs), int(not deferred), (name, deferred, exception))
                    if name != 'qwen35_gdn_finite.py':
                        self.assertEqual(removed, ['original-handle'])
                        self.assertEqual(obj.handles, [])

    def test_staged_restore_calls_original_helper_and_preserves_consume_order(self):
        calls = []
        class Operand:
            def __init__(self, name):
                self.name = name
            def __getitem__(self, _slice):
                return self
        values = {name: Operand(name) for name in ('gate_output', 'up_output',
            'silu_output', 'post_norm_input', 'input_norm_input')}
        def copier(value, destination, **kwargs):
            calls.append(('restore', value.name, destination, kwargs))
            return value
        def mlp(*_args):
            calls.append(('mlp', tuple(values)))
            return 'mnorm'
        def norm(value, *_args):
            calls.append(('norm', value.name, tuple(values)))
            return 'original-norm-coefficients'
        def mixer(_upstream):
            calls.append(('mixer', tuple(values)))
            return 'original-mixer-coefficients', {}
        module = SimpleNamespace(weight=object())
        layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=module, up_proj=module, gate_proj=module),
            post_attention_layernorm=SimpleNamespace(weight=object(), eps=1e-6),
            input_layernorm=SimpleNamespace(weight=object(), eps=1e-6))
        source = self.candidate('qwen35_decoder_finite.py')
        namespace = dict(copy_capture_tensor=copier)
        execute_nodes([top_node(source, '_linear_weights'),
                       top_node(source, 'decoder_finite_pullback')], namespace)
        result = namespace['decoder_finite_pullback'](layer, values, 'upstream', mixer,
            SimpleNamespace(mlp=mlp, norm_residual=norm), consume_captures=True, restore_captures=True)
        self.assertEqual([c[1] for c in calls if c[0] == 'restore'],
            ['gate_output', 'up_output', 'silu_output', 'post_norm_input', 'input_norm_input'])
        self.assertEqual([c[0] for c in calls],
            ['restore', 'restore', 'restore', 'mlp', 'restore', 'norm', 'mixer', 'restore', 'norm'])
        self.assertEqual(tuple(values), ('input_norm_input',))
        self.assertEqual(result, ('original-norm-coefficients', {}))
        for call in calls:
            if call[0] == 'restore':
                self.assertEqual(call[2:], ('cuda', {'preserve_strides': True}))

    def test_restore_false_executes_no_copy_helper(self):
        source = self.candidate('qwen35_decoder_finite.py')
        namespace = dict(copy_capture_tensor=lambda *_a, **_k: self.fail('Default called copier'))
        execute_nodes([top_node(source, '_linear_weights'),
                       top_node(source, 'decoder_finite_pullback')], namespace)
        module = SimpleNamespace(weight=object())
        layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=module, up_proj=module, gate_proj=module),
            post_attention_layernorm=SimpleNamespace(weight=object(), eps=1e-6),
            input_layernorm=SimpleNamespace(weight=object(), eps=1e-6))
        values = {name: [0, 1] for name in ('gate_output', 'up_output', 'silu_output',
                                          'post_norm_input', 'input_norm_input')}
        namespace['decoder_finite_pullback'](layer, values, 'upstream', lambda _: ('mix', {}),
            SimpleNamespace(mlp=lambda *_: 'mlp', norm_residual=lambda *_: 'norm'))


class RealCPUDecoderTransportContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest('Existing CPU Torch is unavailable; not a GPU acceptance result') from error
        cls.torch = torch
        import test_native_root_tape_owner as existing
        cls.fixture = existing.actual_cpu_fixture(torch)
        owner_root, accelerated_root = owner_roots()
        cls.originals = {name: (owner_root / name).read_bytes()
                         for name in preparer.MODULES}
        accelerated_originals = {name: (accelerated_root / name).read_bytes()
                                for name in preparer.ACCELERATED_MODULES}
        candidates, _ = preparer.patched_sources(cls.originals,
            accelerated_originals=accelerated_originals)
        namespace = dict(torch=torch)
        execute_nodes([top_node(cls.originals['native_attention_capture.py'], 'copy_capture_tensor'),
            top_node(candidates[preparer.MODULES['qwen35_decoder_finite.py']], 'NativeDecoderCapture')], namespace)
        cls.Capture = namespace['NativeDecoderCapture']
        original_namespace = dict(torch=torch)
        execute_nodes([top_node(cls.originals['qwen35_decoder_finite.py'], 'NativeDecoderCapture')],
                      original_namespace)
        cls.Original = original_namespace['NativeDecoderCapture']
        retained_namespace = dict(torch=torch, copy_capture_tensor=namespace['copy_capture_tensor'],
                                  _Decoder=cls.Capture, AUDIT=False, AUDIT_RECORDS=[])
        retained_source = candidates[preparer.ACCELERATED_MODULES['qwen35_retained_capture.py']]
        execute_nodes([top_node(retained_source, '_Retain'),
                       top_node(retained_source, 'NativeDecoderCapture')], retained_namespace)
        cls.Retained = retained_namespace['NativeDecoderCapture']

    def test_real_cpu_default_hooks_match_original_values_and_cleanup(self):
        torch, layer = self.torch, self.fixture.Decoder().eval()
        inputs = torch.arange(96, dtype=torch.float32).reshape(8, 3, 4)
        with torch.no_grad(), self.Original(layer, destination='cpu') as reference:
            expected = layer(inputs)
        with torch.no_grad(), self.Capture(layer, destination='cpu') as candidate:
            actual = layer(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(reference.calls, candidate.calls)
        self.assertEqual(set(reference.values), set(candidate.values))
        for name, value in reference.values.items():
            other = candidate.values[name]
            self.assertTrue(torch.equal(value, other), name)
            self.assertEqual(value.dtype, other.dtype)
            self.assertEqual(value.stride(), other.stride())
        self.assertFalse(candidate.handles)

    def test_real_cpu_original_helper_preserves_strides_and_dtype(self):
        torch, layer = self.torch, self.fixture.Decoder().eval()
        for dtype in (torch.float32, torch.bfloat16):
            value = torch.arange(96, dtype=dtype).reshape(8, 3, 4).transpose(1, 2)
            candidate = self.Capture(layer, destination='cpu', preserve_strides=True)
            candidate.retain('actual_view', value)
            actual = candidate.values['actual_view']
            self.assertTrue(torch.equal(value, actual))
            self.assertEqual(value.dtype, actual.dtype)
            self.assertEqual(value.stride(), actual.stride())
            self.assertNotEqual(value.data_ptr(), actual.data_ptr())

    def test_real_cpu_capture_error_removes_original_hooks(self):
        layer = self.fixture.Decoder().eval()
        capture = self.Capture(layer, destination='cpu', preserve_strides=True)
        with self.assertRaisesRegex(RuntimeError, 'actual CPU failure'):
            with capture:
                raise RuntimeError('actual CPU failure')
        self.assertFalse(capture.handles)
        for module in layer.modules():
            self.assertFalse(module._forward_hooks)

    def test_real_cpu_retained_owner_inherits_transport_keywords(self):
        torch, layer = self.torch, self.fixture.Decoder().eval()
        capture = self.Retained(layer, destination='cpu', preserve_strides=True,
                                pinned_host=True, defer_host_sync=True)
        self.assertTrue(capture.preserve_strides)
        self.assertTrue(capture.pinned_host)
        self.assertTrue(capture.defer_host_sync)
        for dtype in (torch.float32, torch.bfloat16):
            value = torch.arange(96, dtype=dtype).reshape(8, 3, 4).transpose(1, 2)
            capture.retain('actual_view', value)
            actual = capture.values['actual_view']
            self.assertTrue(torch.equal(actual, value))
            self.assertEqual(actual.dtype, value.dtype)
            self.assertEqual(actual.stride(), value.stride())
            # Exact original retained owner intentionally borrows same-device
            # inputs with copy=False. This is not a pinned CUDA transfer test.
            self.assertEqual(actual.data_ptr(), value.data_ptr())
        with capture:
            layer(torch.arange(96, dtype=torch.float32).reshape(8, 3, 4))
        self.assertFalse(capture.handles)

    def test_real_cpu_retained_owner_error_removes_original_hooks(self):
        layer = self.fixture.Decoder().eval()
        capture = self.Retained(layer, destination='cpu', preserve_strides=True,
                                pinned_host=True, defer_host_sync=True)
        with self.assertRaisesRegex(RuntimeError, 'actual retained CPU failure'):
            with capture:
                raise RuntimeError('actual retained CPU failure')
        self.assertFalse(capture.handles)
        for module in layer.modules():
            self.assertFalse(module._forward_hooks)


if __name__ == '__main__':
    unittest.main()
