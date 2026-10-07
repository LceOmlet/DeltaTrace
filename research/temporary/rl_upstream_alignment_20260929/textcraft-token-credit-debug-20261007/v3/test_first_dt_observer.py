"""CPU interface tests; no Ray heartbeat, GPU, or attribution accuracy claim."""
import ast
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


v2_tests = load(HERE.parent / 'v2' / 'test_observe_token_credit.py', 'v2_interface_tests')
v2 = v2_tests.module
seam = load(HERE / 'first_dt_observer.py', 'first_dt_observer')


def original_lazy_owner(producer_class, events, rank=0):
    """Execute the exact frozen e5eb original AST, including its host-cache branch.

    CPU tests set VERL_RELEASE_UNUSED_HOST_CACHE=0: that branch is preserved in
    the AST but its CUDA host-cache API and behavior are NOT validated here.
    """
    path = (HERE.parents[1] / 'direct-target-head-memory-20261007' / 'v3' /
            'workload-audit' / 'verl' / 'verl' / 'workers' / 'fsdp_workers.py')
    source = path.read_bytes()
    expected = 'e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39'
    if hashlib.sha256(source).hexdigest() != expected:
        raise AssertionError('CPU interface test requires exact frozen e5eb owner source')
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == 'compute_dt_token_advantages')
    owner_ast = ast.Module(body=[ast.ClassDef(name='OriginalOwner', bases=[], keywords=[],
                                            body=[method], decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(owner_ast)
    namespace = dict(DataProto=object, Dispatch=types.SimpleNamespace(DP_COMPUTE_PROTO=0),
                     register=lambda **kwargs: lambda method: method,
                     load_fsdp_model_to_gpu=lambda actor: events.append('original load'),
                     offload_fsdp_model_to_cpu=lambda actor: events.append('original offload'),
                     os=os, torch=v2.torch)
    exec(compile(owner_ast, str(path), 'exec'), namespace)
    owner = namespace['OriginalOwner']()
    owner.rank, owner._is_offload_param, owner.actor_module_fsdp = rank, True, object()
    owner.config = types.SimpleNamespace(actor={'use_invalid_action_penalty': False})
    owner.update_actor = lambda data: events.append(('original update', rank)) or data
    owner._test_producer_class = producer_class
    return owner


def producer_type(events, *, fail=False):
    class Producer:
        def __init__(self, *args, **kwargs):
            events.append('original producer construction')
            self.direct_readout = v2_tests.Owner()

        def attribute_prepared_batch(self, data):
            events.append('original prepared entry')
            if fail:
                raise ValueError('original prepared failure')
            return self.direct_readout.trajectories([self.direct_readout.item['row']])
    return Producer


def dt_module(producer):
    module = types.ModuleType('deltatrace_rollout')
    module.DeltaTraceRolloutProducer = producer
    return module


class FirstDTTests(unittest.TestCase):
    def setUp(self):
        self.host_cache_setting = patch.dict(os.environ, VERL_RELEASE_UNUSED_HOST_CACHE='0')
        self.host_cache_setting.start()

    def tearDown(self):
        self.host_cache_setting.stop()

    def test_first_original_lazy_call_captured_and_class_restored(self):
        events = []
        producer = producer_type(events)
        original = producer.attribute_prepared_batch
        owner = original_lazy_owner(producer, events)
        worker = types.SimpleNamespace(worker_dict={'actor': owner, 'alias': owner})
        data = types.SimpleNamespace(meta_info={'eos_token_id': 0, 'pad_token_id': 0})
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, deltatrace_rollout=dt_module(producer)):
            record = seam.install_before_first_dt(worker, directory, observer_module=v2, hold=False)
            self.assertEqual(events, [])
            self.assertFalse(hasattr(owner, '_deltatrace_producer'))
            self.assertEqual(len(record['owners']), 1)
            result = owner.compute_dt_token_advantages(data)
            self.assertIs(result, owner._deltatrace_producer.direct_readout.output)
            self.assertIs(producer.attribute_prepared_batch, original)
            self.assertEqual(events, ['original load', 'original producer construction',
                                      'original prepared entry', 'original offload'])
            self.assertTrue((Path(directory) / 'rank0-readout.pt').is_file())
            self.assertTrue((Path(directory) / 'rank0-readout-native-batch-1.pt').is_file())
            sidecar = json.loads((Path(directory) / 'rank0-first-dt-install.json').read_text())
            self.assertEqual(sidecar['status'], 'activated_before_original_prepared_call')
            self.assertEqual(len(sidecar['workers']), 1)
            second = owner.compute_dt_token_advantages(data)
            self.assertIs(second, result)
            self.assertEqual(events.count('original producer construction'), 1)
            self.assertEqual(events.count('original prepared entry'), 2)

    def test_original_prepared_exception_and_FSDP_finally_unchanged(self):
        events = []
        producer = producer_type(events, fail=True)
        original = producer.attribute_prepared_batch
        owner = original_lazy_owner(producer, events)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, deltatrace_rollout=dt_module(producer)):
            seam.install_before_first_dt(types.SimpleNamespace(worker_dict={'actor': owner}),
                                         directory, observer_module=v2, hold=False)
            with self.assertRaisesRegex(ValueError, 'original prepared failure'):
                owner.compute_dt_token_advantages(types.SimpleNamespace(
                    meta_info={'eos_token_id': 0, 'pad_token_id': 0}))
            self.assertIs(producer.attribute_prepared_batch, original)
            self.assertEqual(events[-1], 'original offload')

    def test_unrelated_instance_transparent_does_not_consume_first_DT_hook(self):
        events = []
        producer = producer_type(events)
        owner = original_lazy_owner(producer, events)
        with tempfile.TemporaryDirectory() as directory:
            record = seam.install_before_first_dt(types.SimpleNamespace(worker_dict={'actor': owner}),
                                                 directory, observer_module=v2, producer_class=producer,
                                                 hold=False)
            unrelated = producer()
            result = unrelated.attribute_prepared_batch(None)
            self.assertIs(result, unrelated.direct_readout.output)
            self.assertEqual(record['status'], 'armed_not_yet_activated')
            self.assertFalse((Path(directory) / 'rank0-readout.pt').exists())
            with patch.dict(sys.modules, deltatrace_rollout=dt_module(producer)):
                owner.compute_dt_token_advantages(types.SimpleNamespace(
                    meta_info={'eos_token_id': 0, 'pad_token_id': 0}))
            self.assertTrue((Path(directory) / 'rank0-readout.pt').is_file())

    def test_two_rank_release_paths_unique_and_original_updates_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            owners, paths, events = [], [], []
            for rank in range(2):
                producer = producer_type(events)  # Real ranks have separate processes/classes.
                owner = original_lazy_owner(producer, events, rank)
                owners.append(owner)
                record = seam.install_before_first_dt(types.SimpleNamespace(worker_dict={'actor': owner}),
                                                     directory, observer_module=v2, producer_class=producer)
                paths.append(Path(record['owners'][0]['release_file']))
                with patch.dict(sys.modules, deltatrace_rollout=dt_module(producer)):
                    owner.compute_dt_token_advantages(types.SimpleNamespace(
                        meta_info={'eos_token_id': 0, 'pad_token_id': 0}))
            self.assertNotEqual(paths[0], paths[1])
            data = types.SimpleNamespace(batch={}, non_tensor_batch={})
            threads = [threading.Thread(target=owner.update_actor, args=(data,)) for owner in owners]
            for thread in threads:
                thread.start()
            try:
                deadline = time.monotonic() + 5
                while not all((Path(directory) / f'rank{rank}-hold.json').is_file()
                              for rank in range(2)) and time.monotonic() < deadline:
                    threading.Event().wait(.01)
                self.assertTrue(all(thread.is_alive() for thread in threads))
                self.assertFalse(any(isinstance(item, tuple) for item in events))
                paths[0].touch()
                threads[0].join(5)
                self.assertFalse(threads[0].is_alive())
                self.assertTrue(threads[1].is_alive())
                self.assertEqual([item for item in events if isinstance(item, tuple)], [('original update', 0)])
            finally:
                for path in paths:
                    path.touch()
                for thread in threads:
                    thread.join(5)
            self.assertEqual([item for item in events if isinstance(item, tuple)],
                             [('original update', 0), ('original update', 1)])


def load_tests(loader, tests, pattern):
    tests.addTests(loader.loadTestsFromModule(v2_tests))
    return tests


if __name__ == '__main__':
    unittest.main()
