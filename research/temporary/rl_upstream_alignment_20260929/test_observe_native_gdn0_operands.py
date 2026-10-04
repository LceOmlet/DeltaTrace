"""Passive-event transport contracts only; no toy FLA numerical acceptance."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

from observe_native_gdn0_operands import NativeGDN0Operands, check_saved_fla


class EventInterfaceSpy:
    """Only observe the event seam; never implement a mixer or recurrence."""
    def __init__(self, module):
        self.module = module
        self.codes = {'module': 'module', 'fla': 'FLA', 'stage': 'stage'}
        self.active = True
        self.endpoints = {'o': None}
        self.calls = {}
        self.coefficient_start = 0
        self.original_events = []

    def event(self, frame, kind, value):
        self.original_events.append((frame.f_code, kind, value))


class PassiveEventContracts(unittest.TestCase):
    def setUp(self):
        self.target = object()
        self.audit = NativeGDN0Operands(self.target, 'unused-diagnostic-directory',
                                       variant='off-replay', rank=0)
        self.saved = []
        self.snapshots = []
        self.audit._metadata = lambda value: {'spy_identity': id(value)}
        def observe_copy(value):
            self.snapshots.append(value)
            return value
        self.audit._snapshot = observe_copy
        self.audit._save = self.saved.append
        self.Capture = self.audit.capture_type(EventInterfaceSpy)

    @staticmethod
    def frame(label, **fields):
        return SimpleNamespace(f_code=label, f_locals=fields)

    def test_actual_call_and_stage_initial_state_are_preserved(self):
        capture = self.Capture(self.target)
        capture.event(self.frame('module', self=self.target), 'call', None)
        raw = {name: object() for name in ('q', 'k', 'v', 'g', 'beta')}
        initial = object()
        fla = dict(**raw, initial_state=initial, cu_seqlens=None,
                   scale=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
        capture.event(self.frame('fla', **fla), 'call', None)
        stage = dict(initial_state=initial, cu_seqlens=None,
                     scale=0.125, output_final_state=True)
        capture.event(self.frame('stage', **stage), 'call', None)
        actual_stage = {name: object() for name in ('o', 'final_state', 'k', 'w', 'u', 'g', 'h')}
        capture.event(self.frame('stage', **actual_stage), 'return', object())
        output, final_state = object(), object()
        capture.event(self.frame('fla'), 'return', (output, final_state))
        capture.event(self.frame('module', self=self.target), 'return', object())
        self.assertEqual(len(self.saved), 1)
        pending = self.saved[0]
        self.assertIs(pending['tensors']['initial_state'], initial)
        self.assertIs(pending['tensors']['fla_initial_state'], initial)
        self.assertIsNone(pending['tensors']['cu_seqlens'])
        self.assertIs(pending['tensors']['raw_q'], raw['q'])
        self.assertIs(pending['tensors']['raw_k'], raw['k'])
        self.assertIs(pending['tensors']['native_o'], output)
        self.assertIs(pending['tensors']['native_ht'], final_state)
        self.assertEqual(pending['calls']['stage']['scale'], 0.125)
        self.assertIsNone(pending['calls']['fla']['scale'])
        self.assertTrue(pending['calls']['fla']['use_qk_l2norm_in_kernel'])
        self.assertEqual(len(capture.original_events), 6)
        self.assertFalse(hasattr(capture, '_gdn0_observation'))

    def test_unrelated_original_capture_is_only_forwarded(self):
        capture = self.Capture(object())
        event = self.frame('stage')
        marker = object()
        capture.event(event, 'return', marker)
        self.assertEqual(capture.original_events, [('stage', 'return', marker)])
        self.assertFalse(self.saved or self.snapshots)

    def test_original_event_runs_before_observation_and_exception_is_preserved(self):
        class RaisingOwner(EventInterfaceSpy):
            def event(self, frame, kind, value):
                raise ValueError('original event failure')
        capture = self.audit.capture_type(RaisingOwner)(self.target)
        with self.assertRaisesRegex(ValueError, 'original event failure'):
            capture.event(self.frame('module', self=self.target), 'call', None)
        self.assertFalse(self.saved or self.snapshots)

    def test_subclass_preserves_original_constructor_and_only_overrides_event(self):
        self.assertIs(self.Capture.__init__, EventInterfaceSpy.__init__)
        methods = {name for name, value in self.Capture.__dict__.items()
                   if inspect.isfunction(value)}
        self.assertEqual(methods, {'event'})
        self.assertEqual(len(self.audit.report()['records']), 0)

    def test_reference_reads_actual_initial_state_and_original_assertions(self):
        parsed = ast.parse(inspect.getsource(check_saved_fla))
        calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)]
        reference = next(node for node in calls if isinstance(node.func, ast.Attribute)
                         and node.func.attr == 'recurrent_gated_delta_rule_ref')
        initial = next(keyword.value for keyword in reference.keywords
                       if keyword.arg == 'initial_state')
        self.assertEqual(ast.unparse(initial), "move(tensors['initial_state'])")
        source = inspect.getsource(check_saved_fla)
        self.assertIn('exec(assertion_code', source)
        self.assertIn('owner.FLA_CI_ENV', source)
        self.assertNotIn('initial_state=None', source)
        self.assertNotIn('0.005', source)
        self.assertEqual(source.count('F.normalize(q,'), 1)
        self.assertEqual(source.count('F.normalize(k,'), 1)

    def test_observation_never_calls_model_operator_or_global_profile(self):
        parsed = ast.parse(inspect.getsource(NativeGDN0Operands))
        names = set()
        for call in (node for node in ast.walk(parsed) if isinstance(node, ast.Call)):
            names.add(call.func.id if isinstance(call.func, ast.Name) else
                      call.func.attr if isinstance(call.func, ast.Attribute) else None)
        self.assertFalse(names & {'forward', 'model', 'layer', 'chunk_gated_delta_rule',
                                  'recurrent_gated_delta_rule_ref', 'setprofile', 'synchronize'})


class ActualCPUTensorTransport(unittest.TestCase):
    def test_existing_dtype_shape_and_immutable_cpu_snapshot(self):
        try:
            import torch
        except ImportError as error:
            self.skipTest(f'Existing CPU Torch unavailable: {error}')
        original = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
        saved = NativeGDN0Operands._snapshot(original)
        self.assertTrue(torch.equal(original, saved))
        self.assertEqual(saved.dtype, original.dtype)
        self.assertEqual(saved.shape, original.shape)
        original.add_(1)
        self.assertFalse(torch.equal(original, saved))
        self.assertEqual(NativeGDN0Operands._metadata(original)['stride'], list(original.stride()))
        self.assertIsNone(NativeGDN0Operands._snapshot(None))


if __name__ == '__main__':
    unittest.main()
