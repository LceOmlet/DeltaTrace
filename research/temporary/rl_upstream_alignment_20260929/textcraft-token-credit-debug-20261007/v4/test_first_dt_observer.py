"""Prepared-only CPU interfaces, not GPU numerical or Ray heartbeat evidence."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


v3_tests = load(HERE.parent / 'v3' / 'test_first_dt_observer.py', 'v3_original_interface_tests')
seam = load(HERE / 'first_dt_observer.py', 'first_dt_observer_v4')
v3_tests.seam = seam
v2 = v3_tests.v2


class FirstDTTests(v3_tests.FirstDTTests):
    """Inherit all v3 tests, including the exact frozen e5eb method AST."""

    def test_complete_prepared_data_saved_before_original_exception(self):
        events = []
        producer = v3_tests.producer_type(events, fail=True)
        original = producer.attribute_prepared_batch
        owner = v3_tests.original_lazy_owner(producer, events)
        data = types.SimpleNamespace(
            batch={'input_ids': v2.torch.tensor([[11, 12, 13]]),
                   'policy_mask': v2.torch.tensor([[False, True, True]]),
                   'target_mask': v2.torch.tensor([[False, False, True]]),
                   'dt_direct_reward': v2.torch.tensor([.8], dtype=v2.torch.float64),
                   'extra_original_tensor': v2.torch.tensor([7], dtype=v2.torch.int16)},
            non_tensor_batch={'traj_uid': ['native-joint'], 'offsets': [[2]],
                              'original_metadata': {'unmodified': True}},
            meta_info={'eos_token_id': 0, 'pad_token_id': 0, 'original_option': [1, 2]})
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, deltatrace_rollout=v3_tests.dt_module(producer)):
            path = Path(directory) / 'rank0-input-prepared.pt'

            def prepared(instance, received):
                self.assertIs(received, data)
                self.assertTrue(path.is_file())
                return original(instance, received)

            producer.attribute_prepared_batch = prepared
            seam.install_before_first_dt(types.SimpleNamespace(worker_dict={'actor': owner}),
                                         directory, observer_module=v2, hold=False)
            with self.assertRaisesRegex(ValueError, 'original prepared failure'):
                owner.compute_dt_token_advantages(data)
            saved = v2.torch.load(path, weights_only=False)
            self.assertEqual(set(saved['batch']), set(data.batch))
            for key, expected in data.batch.items():
                self.assertTrue(v2.torch.equal(saved['batch'][key], expected))
                self.assertEqual(saved['batch'][key].dtype, expected.dtype)
                self.assertEqual(saved['batch'][key].device.type, 'cpu')
            self.assertEqual(saved['non_tensor_batch'], data.non_tensor_batch)
            self.assertEqual(saved['meta_info'], data.meta_info)
            self.assertEqual(events.count('original prepared entry'), 1)
            self.assertEqual(events[-1], 'original offload')
            self.assertIs(producer.attribute_prepared_batch, prepared)

    def test_optional_scope_preserves_same_owner_result_and_finally(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                events = []
                producer = v3_tests.producer_type(events, fail=fail)
                original = producer.attribute_prepared_batch
                owner = v3_tests.original_lazy_owner(producer, events)

                @contextmanager
                def scope(received_owner, received_producer):
                    self.assertIs(received_owner, owner)
                    self.assertIs(received_producer, owner._deltatrace_producer)
                    self.assertTrue((Path(directory) / 'rank0-input-prepared.pt').is_file())
                    events.append('passive scope entered')
                    try:
                        yield
                    finally:
                        events.append('passive scope restored')

                record = seam.install_before_first_dt(
                    types.SimpleNamespace(worker_dict={'actor': owner}), directory,
                    observer_module=v2, hold=False, replay_scope_factory=scope,
                    producer_class=producer)
                with patch.dict(sys.modules, deltatrace_rollout=v3_tests.dt_module(producer)):
                    data = types.SimpleNamespace(batch={}, non_tensor_batch={},
                        meta_info={'eos_token_id': 0, 'pad_token_id': 0})
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'original prepared failure'):
                            owner.compute_dt_token_advantages(data)
                    else:
                        result = owner.compute_dt_token_advantages(data)
                        self.assertIs(result, owner._deltatrace_producer.direct_readout.output)
                self.assertIs(producer.attribute_prepared_batch, original)
                self.assertEqual(events.count('original prepared entry'), 1)
                self.assertEqual(events[-2:], ['passive scope restored', 'original offload'])
                self.assertTrue(record['replay_scope_enabled'])

    def test_snapshot_error_does_not_replace_original_exception(self):
        events = []
        producer = v3_tests.producer_type(events, fail=True)
        owner = v3_tests.original_lazy_owner(producer, events)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, deltatrace_rollout=v3_tests.dt_module(producer)):
            record = seam.install_before_first_dt(
                types.SimpleNamespace(worker_dict={'actor': owner}), directory,
                observer_module=v2, hold=False)
            with patch.object(v2.torch, 'save', side_effect=OSError('diagnostic disk error')):
                with self.assertRaisesRegex(ValueError, 'original prepared failure'):
                    owner.compute_dt_token_advantages(types.SimpleNamespace(
                        batch={}, non_tensor_batch={},
                        meta_info={'eos_token_id': 0, 'pad_token_id': 0}))
            self.assertFalse(record['prepared_inputs'][0]['saved'])
            self.assertIn('diagnostic disk error', record['prepared_inputs'][0]['observer_error'])
            self.assertEqual(events[-1], 'original offload')


def load_tests(loader, tests, pattern):
    tests.addTests(loader.loadTestsFromModule(v3_tests.v2_tests))
    return tests


if __name__ == '__main__':
    unittest.main()
