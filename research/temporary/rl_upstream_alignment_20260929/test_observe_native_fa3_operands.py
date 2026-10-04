"""CPU transport/AST contracts only; no fake FA numerical acceptance.

The small metadata objects observe copy calls, not tensor or attention math.
Actual pinned source AST can additionally be checked when the saved official
file is supplied through DT_OFFICIAL_FA_TEST_SOURCE on the provisioned host.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HELPER = HERE / 'observe_native_fa3_operands.py'
SPEC = importlib.util.spec_from_file_location('_fa3_passive_boundary', HELPER)
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


class MetadataOperand:
    def __init__(self, shape, log):
        self.shape, self.log = shape, log
        self.dtype, self.device = 'torch.bfloat16', 'cuda:0'

    def __getitem__(self, _key):
        raise AssertionError('Complete actual operands must never be sliced.')

    def detach(self):
        self.log.append('detach')
        return self

    def to(self, device, *, copy):
        self.log.append(('copy', device, copy))
        return object()

    def stride(self):
        b, t, h, d = self.shape
        return (t * h * d, h * d, d, 1)

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self):
        return 2


class OriginalCaptureBoundary:
    """Exit lifecycle spy only; no forward, capture or FA implementation."""
    def __init__(self, module, *, exit_result=None, exit_error=None):
        self.module, self.exit_result, self.exit_error = module, exit_result, exit_error
        self.events = []
        self.values = {}
        shapes = {'dense_q': (8, 576, 16, 256), 'dense_k': (8, 7552, 4, 256),
                  'dense_v': (8, 7552, 4, 256), 'attention_output': (8, 576, 16, 256)}
        for name, shape in shapes.items():
            self.values[name] = MetadataOperand(shape, self.events)
        self.dense_arguments = {'causal': True, 'dropout_p': 0., 'softmax_scale': .0625,
                                'return_attn_probs': False}
        self.calls = {'module': 1, 'interface': 1, 'native_varlen': 0, 'native_dense': 1}

    def __exit__(self, *exc_info):
        self.events.append(('original_exit', exc_info))
        if self.exit_error is not None:
            raise self.exit_error
        return self.exit_result


class PassiveFA3TransportContracts(unittest.TestCase):
    def observed(self, target, records, **kwargs):
        cls = OWNER.make_attention_capture_observer(
            OriginalCaptureBoundary, target_module=target, records=records)
        return cls(target, **kwargs)

    def test_original_exit_then_complete_operands_and_exact_return(self):
        target, returned, records = object(), object(), []
        capture = self.observed(target, records, exit_result=returned)
        actual = capture.__exit__(None, None, None)
        self.assertIs(actual, returned)
        self.assertEqual(capture.events[0][0], 'original_exit')
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual(set(row['tensors']), set(OWNER.SNAPSHOT_FIELDS))
        self.assertEqual(row['source_metadata']['dense_q']['shape'], [8, 576, 16, 256])
        self.assertEqual(row['source_metadata']['dense_k']['shape'], [8, 7552, 4, 256])
        self.assertEqual(row['source_metadata']['dense_v']['shape'], [8, 7552, 4, 256])
        self.assertEqual(row['source_metadata']['attention_output']['shape'], [8, 576, 16, 256])
        self.assertEqual(capture.events[1:], ['detach', ('copy', 'cpu', True)] * 4)
        self.assertEqual(row['native_calls'], capture.calls)
        self.assertEqual(row['dense_arguments'], capture.dense_arguments)
        self.assertIsNot(row['dense_arguments'], capture.dense_arguments)
        self.assertEqual(row['model_forward_calls_added'], 0)
        self.assertEqual(row['snapshot_copies'], 4)
        self.assertTrue(all(isinstance(value, MetadataOperand) for value in capture.values.values()))

    def test_other_module_runs_original_exit_without_snapshot(self):
        records, target = [], object()
        cls = OWNER.make_attention_capture_observer(
            OriginalCaptureBoundary, target_module=target, records=records)
        capture = cls(object())
        capture.__exit__(None, None, None)
        self.assertEqual(capture.events, [('original_exit', (None, None, None))])
        self.assertFalse(records)

    def test_original_forward_error_does_not_snapshot(self):
        records = []
        capture = self.observed(object(), records)
        error = ValueError('original forward error')
        capture.__exit__(ValueError, error, None)
        self.assertEqual(capture.events, [('original_exit', (ValueError, error, None))])
        self.assertFalse(records)

    def test_original_exit_error_is_preserved_without_snapshot(self):
        records = []
        error = RuntimeError('original capture cleanup error')
        capture = self.observed(object(), records, exit_error=error)
        with self.assertRaises(RuntimeError) as caught:
            capture.__exit__(None, None, None)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(capture.events), 1)
        self.assertFalse(records)

    def test_helper_has_no_model_or_flash_forward_and_no_tensor_slicing(self):
        tree = ast.parse(HELPER.read_text(encoding='utf8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                 for n in calls if isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertFalse(names & {'forward', 'model', 'layer', 'flash_attn_func',
                                  'decoder_finite_pullback', 'gdn_finite_pullback'})
        self.assertFalse(any(isinstance(n, ast.Slice) for n in ast.walk(tree)))
        check = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'check_saved_operands')
        # Numerical thresholds are exclusively original source assertions.
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(check)))
        reference_calls = [n for n in ast.walk(check) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Subscript)
                           and isinstance(n.func.slice, ast.Constant)
                           and n.func.slice.value == 'attention_ref']
        self.assertEqual(len(reference_calls), 2)
        self.assertTrue(all(next(k.value.value for k in n.keywords if k.arg == 'causal')
                            for n in reference_calls))

    def test_fixed_official_source_hash_rejects_unrelated_source(self):
        with tempfile.TemporaryDirectory(prefix='fa3-source-boundary-') as directory:
            source = Path(directory) / 'not-the-fixed-official-test.py'
            source.write_bytes(b'pass\n')
            with self.assertRaisesRegex(ValueError, 'Fixed FA test source SHA256 mismatch'):
                OWNER.official_source_nodes(source)

    def test_pinned_official_reference_and_output_assertion_ast_are_unchanged(self):
        name = os.environ.get('DT_OFFICIAL_FA_TEST_SOURCE')
        if not name:
            self.skipTest('Pinned official FA source is saved on the provisioned host')
        source = Path(name)
        functions, assertions, provenance = OWNER.official_source_nodes(source)
        original = ast.parse(source.read_bytes().decode('utf8'))
        refs = [n for n in original.body if isinstance(n, ast.FunctionDef)
                and n.name in ('attention_ref', 'construct_local_mask')]
        test = next(n for n in original.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'test_flash_attn_output')
        checks = [n for n in test.body if isinstance(n, ast.Assert)
                  and 'out_ref' in ast.unparse(n)]
        normalize = lambda nodes: ast.dump(ast.Module(body=nodes, type_ignores=[]),
                                          include_attributes=False)
        self.assertEqual(normalize(functions), normalize(refs))
        self.assertEqual(normalize(assertions), normalize(checks))
        self.assertEqual(provenance['sha256'], OWNER.OFFICIAL_FA_SHA256)


if __name__ == '__main__':
    unittest.main()
