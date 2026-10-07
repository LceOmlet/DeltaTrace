"""CPU-only interface tests: no model/DT/reward numerical validation claims."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import tarfile
import types
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('token_credit_observer', HERE / 'observe_token_credit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trace_token_attribution(*args, **kwargs):
    """Already-computed CPU owner result; no model or credit formula fixture."""
    return (torch.tensor([[.25, -4., 0., .5, 0., 9.]], dtype=torch.float64), None, {})


def item_fixture():
    row = dict(input_ids=torch.tensor([0, 11, 21, 0, 22, 23, 24]),
               attention_mask=torch.tensor([0, 1, 1, 0, 1, 1, 1]),
               responses=torch.tensor([21, 0, 22, 23, 24]),
               policy_mask=torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool),
               target_mask=torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool),
               dt_direct_reward=1., traj_uid='native-uid')
    return dict(index=0, row=row, width=5, selected=torch.tensor([11, 21, 22, 23, 24]),
                suffix_positions=torch.tensor([0, 2, 3, 4]), prompt_length=1,
                policy=row['policy_mask'], target=row['target_mask'],
                prior=torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool),
                valid_policy=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
                valid_target=torch.tensor([0, 1, 1, 0], dtype=torch.bool),
                target_offsets=[1, 2], reward=1., ratios=torch.zeros(5),
                case={'target_ids': torch.tensor([21, 22, 23, 24]), 'prompt_length': 1})


class Owner:
    def __init__(self):
        self.item = item_fixture()
        self.output = [dict(dt_token_advantages=torch.tensor([-53., 0., 1., 1., 0.]),
                            dt_q_estimates=torch.tensor([1., 0., 1., 1., 0.]),
                            dt_v_estimates=torch.tensor([54., 0., 0., 0., 0.]))]
        self.last_report = {'traces': [dict(trajectory_index=0,
                                           factual_target_logp=-5., reference_target_logp=-6.)]}
        self.fail = False

    def _prepare_row(self, row, index):
        return self.item

    @torch.no_grad()
    def trajectories(self, rows):
        item = self._prepare_row(rows[0], 0)
        if self.fail:
            raise ValueError('original owner failure')
        native = trace_token_attribution(None, None, None, [item['case']], [item['target_offsets']])
        self.returned_trace = native
        item['ratios'][0] = native[0][0, 1].float()
        return self.output


class ObserverTests(unittest.TestCase):
    def test_default_is_inert(self):
        owner = Owner()
        self.assertNotIn('trajectories', vars(owner))
        self.assertNotIn('_prepare_row', vars(owner))

    def test_exact_objects_dtype_masks_and_three_positions(self):
        owner, original_trace = Owner(), trace_token_attribution
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'readout.pt'
            observation = module.ReadoutObservation(owner, path, provenance={'source': 'frozen'}).install()
            result = owner.trajectories([owner.item['row']])
            self.assertIs(result, owner.output)
            self.assertIs(trace_token_attribution, original_trace)
            self.assertTrue(observation.saved)
            self.assertTrue(observation.restored)
            self.assertNotIn('trajectories', vars(owner))
            self.assertNotIn('_prepare_row', vars(owner))
            saved = torch.load(path, weights_only=False)
            row = saved['rows'][0]
            torch.testing.assert_close(row['native_signed_packed'], owner.returned_trace[0][0])
            self.assertEqual(row['native_signed_dtype'], 'torch.float64')
            self.assertEqual(row['ratios_dtype'], 'torch.float32')
            self.assertEqual(row['response_to_packed'].tolist(), [1, -1, 2, 3, 4])
            self.assertEqual(row['target_offsets'], [1, 2])
            self.assertEqual(row['ratios'][2:4].tolist(), [0., 0.])
            self.assertFalse(row['self_target_branch']['finite_d_measured'])
            extreme = row['extrema']['dt_token_advantages']['prior_source']['min']
            self.assertEqual((extreme['response_slot'], extreme['original_input_slot'],
                              extreme['packed_input_slot'], extreme['token_id']), (0, 2, 1, 21))
            self.assertEqual(row['extrema']['dt_token_advantages']['self_target']['max']['target_predictor_slot'], 1)
            self.assertEqual(row['extrema']['ratios']['masked']['tokens'], 2)
            self.assertEqual(row['trace']['factual_target_logp'], -5.)
            self.assertEqual(saved['provenance']['source'], 'frozen')
            torch.testing.assert_close(row['outputs']['dt_token_advantages'], result[0]['dt_token_advantages'])

    def test_original_exception_and_bindings_restored(self):
        owner, original_trace = Owner(), trace_token_attribution
        owner.fail = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'readout.pt'
            observation = module.ReadoutObservation(owner, path).install()
            with self.assertRaisesRegex(ValueError, 'original owner failure'):
                owner.trajectories([owner.item['row']])
            self.assertTrue(observation.restored)
            self.assertIs(trace_token_attribution, original_trace)
            self.assertFalse(path.exists())

    def test_save_failure_does_not_change_owner_output(self):
        owner = Owner()
        with tempfile.TemporaryDirectory() as directory:
            observation = module.ReadoutObservation(owner, directory).install()
            self.assertIs(owner.trajectories([owner.item['row']]), owner.output)
            self.assertFalse(observation.saved)
            self.assertEqual(observation.errors[-1]['stage'], 'save_readout')
            self.assertTrue(observation.restored)

    def test_worker_dispatch_capture_hold_before_original_update(self):
        readout = Owner()
        events = []

        class Worker:
            rank = 1
            _deltatrace_producer = SimpleNamespace(direct_readout=readout)

            def update_actor(self, data):
                events.append('original update')
                return data

        worker = Worker()
        data = SimpleNamespace(batch={'advantages': torch.tensor([[2., -3.]]),
                                      'responses': torch.tensor([[21, 23]]),
                                      'loss_mask': torch.tensor([[1, 1]], dtype=torch.bool)},
                               non_tensor_batch={'uid': ['native-uid']})
        with tempfile.TemporaryDirectory() as directory:
            records = module.install_on_worker_dict(
                SimpleNamespace(worker_dict={'actor': worker, 'alias': worker}),
                directory, provenance={'pid_birth': 10.})
            self.assertEqual(len(records), 1)
            readout.trajectories([readout.item['row']])
            # Official WorkerDict uses this dynamic getattr rather than a cached method.
            returned = []
            thread = threading.Thread(target=lambda: returned.append(getattr(worker, 'update_actor')(data)))
            thread.start()
            release = Path(records[0]['release_file'])
            sidecar = Path(records[0]['hold_sidecar'])
            try:
                deadline = time.monotonic() + 5
                while not sidecar.exists() and time.monotonic() < deadline:
                    threading.Event().wait(.01)
                self.assertTrue(sidecar.exists())
                self.assertNotIn('update_actor', vars(worker))
                self.assertEqual(events, [])
                self.assertTrue(thread.is_alive())
                # The test's main thread remains runnable while the real wait holds.
                marker = Path(directory) / 'main-thread-running'
                marker.write_text('alive')
                self.assertEqual(marker.read_text(), 'alive')
                hold = json.loads(sidecar.read_text())
                self.assertTrue(hold['hold_entered'])
                self.assertEqual(hold['release_file'], str(release))
            finally:
                release.touch()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(returned), 1)
            self.assertIs(returned[0], data)
            self.assertEqual(events, ['original update'])
            saved = torch.load(Path(directory) / 'rank1-pre-update.pt', weights_only=False)
            torch.testing.assert_close(saved['tensors']['advantages'], data.batch['advantages'])
            self.assertEqual(saved['non_tensors']['uid'], ['native-uid'])
            self.assertIn('dt_direct_target_artifact', saved['missing_non_tensor_fields'])
            self.assertTrue(records[0]['hold_entered'])

    def test_save_errors_still_hold_until_explicit_release(self):
        calls = []

        class Worker:
            rank = 0
            _deltatrace_producer = SimpleNamespace(direct_readout=Owner())

            def update_actor(self, data):
                calls.append(data)
                return data

        with tempfile.TemporaryDirectory() as directory:
            worker = Worker()
            output = Path(directory) / 'not-a-directory'
            output.write_text('intentional observation I/O failure')
            release = Path(directory) / 'explicit-release'
            records = module.install_on_worker_dict(SimpleNamespace(worker_dict={'actor': worker}),
                                                    output, release_file=release)
            data = SimpleNamespace(batch={}, non_tensor_batch={})
            thread = threading.Thread(target=worker.update_actor, args=(data,))
            thread.start()
            try:
                deadline = time.monotonic() + 5
                while not records[0].get('hold_entered') and time.monotonic() < deadline:
                    threading.Event().wait(.01)
                self.assertTrue(records[0]['hold_entered'])
                self.assertIn('observation_error', records[0])
                self.assertIn('hold_sidecar_error', records[0])
                self.assertEqual(calls, [])
                self.assertTrue(thread.is_alive())
            finally:
                release.touch()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(calls, [data])

    def test_completed_native_batch_survives_later_original_trace_failure(self):
        calls = []
        original_trace = trace_token_attribution

        def later_failure(*args, **kwargs):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError('original second trace failure')
            return original_trace(*args, **kwargs)

        class TwoBatchOwner(Owner):
            @torch.no_grad()
            def trajectories(self, rows):
                item = self._prepare_row(rows[0], 0)
                trace_token_attribution(None, None, None, [item['case']], [item['target_offsets']])
                trace_token_attribution(None, None, None, [item['case']], [item['target_offsets']])
                return self.output

        owner = TwoBatchOwner()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                globals(), trace_token_attribution=later_failure):
            path = Path(directory) / 'rank0-readout.pt'
            observation = module.ReadoutObservation(owner, path).install()
            with self.assertRaisesRegex(RuntimeError, 'original second trace failure'):
                owner.trajectories([owner.item['row']])
            self.assertTrue(observation.restored)
            self.assertIs(trace_token_attribution, later_failure)
            self.assertFalse(path.exists())  # No fabricated final Q/V/A output.
            batch = torch.load(Path(directory) / 'rank0-readout-native-batch-1.pt', weights_only=False)
            self.assertEqual(batch['native_signed_dtype'], 'torch.float64')
            torch.testing.assert_close(batch['native_signed'], original_trace()[0])
            self.assertEqual(batch['rows'][0]['trajectory_index'], 0)
            self.assertEqual(batch['rows'][0]['target_offsets'], [1, 2])
            self.assertEqual(batch['rows'][0]['suffix_positions'].tolist(), [0, 2, 3, 4])
            self.assertFalse((Path(directory) / 'rank0-readout-native-batch-2.pt').exists())

    def test_frozen_original_readout_sort_b4_duplicate_uid_and_exact_qva(self):
        """Real deployed owner + real Q/V composition, CPU trace transport marker.

        This tests observation identity/alignment, NOT DT/model numeric accuracy.
        The marker never implements attribution or rewards; the frozen owner
        exclusively performs preparation, sorting, batching, scatter, and Q/V/A.
        """
        archive = HERE.parents[1] / 'direct-action-target-20261007' / 'direct-overlay-v3.tar'
        expected = {'reward_readout.py': '31e2acfb760eb1dd118af78a5a887898357feacc1b0f1caec04f06006a3b7d07',
                    'counterfactual.py': '0d3412b8ac25d7c036e7a827d4bbf19a4eb2b9984a0a0956f37e82d7d86f9d07'}
        source = {}
        with tarfile.open(archive) as bundle:
            for name, digest in expected.items():
                member = next(entry for entry in bundle.getmembers() if entry.name.endswith(name))
                source[name] = bundle.extractfile(member).read()
                self.assertEqual(hashlib.sha256(source[name]).hexdigest(), digest)
        calls = []

        def marker(runner, reference, factual, target_case, target_offsets, **kwargs):
            calls.append([case['prompt_length'] for case in target_case])
            signed = torch.zeros_like(factual, dtype=torch.float64)
            for index, case in enumerate(target_case):
                signed[index, case['prompt_length']] = .123456789123 + case['prompt_length'] / 100.
            audit = dict(root_effect=.1, policy_credit_signed_sum=.1, conservation_residual=0.,
                         factual_target_logp=-5., reference_target_logp=-6.,
                         complete_attribution_seconds_with_diagnostics=0.)
            detail = dict(audit, per_sample=[audit.copy() for _ in target_case])
            return signed, None, detail

        counterfactual = types.ModuleType('counterfactual')
        transport = types.ModuleType('deltatrace_credit')
        transport.trace_token_attribution = marker
        readout_module = types.ModuleType('frozen_original_readout_for_cpu_observer_test')
        with patch.dict(sys.modules, counterfactual=counterfactual,
                        deltatrace_credit=transport,
                        frozen_original_readout_for_cpu_observer_test=readout_module):
            exec(compile(source['counterfactual.py'], '<frozen counterfactual 0d3412>', 'exec'),
                 counterfactual.__dict__)
            exec(compile(source['reward_readout.py'], '<frozen readout 31e2ac>', 'exec'),
                 readout_module.__dict__)
            rows = []
            for extra in [3, 0, 5, 1, 4, 2]:
                row = item_fixture()['row']
                row['input_ids'] = torch.cat((torch.tensor([0, 11] + [12] * extra), row['responses']))
                row['attention_mask'] = torch.cat((torch.tensor([0, 1] + [1] * extra),
                                                   torch.tensor([1, 0, 1, 1, 1])))
                rows.append(row)  # Duplicate UIDs intentionally remain distinct cases.
            kwargs = dict(task='TextCraft', packed_answer_targets=None, minibatch_size=4)
            runner = SimpleNamespace(model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros(1))))
            baseline = readout_module.DirectActionTargetReadout(runner, SimpleNamespace(eos_token_id=0), **kwargs)
            observed = readout_module.DirectActionTargetReadout(runner, SimpleNamespace(eos_token_id=0), **kwargs)
            original_values = baseline.trajectories(rows)
            baseline_calls = calls.copy()
            calls.clear()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'frozen-owner.pt'
                observation = module.ReadoutObservation(observed, path).install()
                observed_values = observed.trajectories(rows)
                self.assertTrue(observation.saved, observation.errors)
                self.assertEqual(calls, baseline_calls)
                self.assertEqual(calls, [[1, 2, 3, 4], [5, 6]])
                self.assertIs(readout_module.trace_token_attribution, marker)
                for before, after in zip(original_values, observed_values):
                    for key in before:
                        torch.testing.assert_close(after[key], before[key], rtol=0., atol=0.)
                saved = torch.load(path, weights_only=False)
                self.assertEqual(len(saved['rows']), len(rows))
                for index, row in enumerate(saved['rows']):
                    self.assertEqual(row['trajectory_index'], index)
                    self.assertEqual(row['native_signed_dtype'], 'torch.float64')
                    self.assertEqual(row['ratios_dtype'], 'torch.float32')
                    self.assertEqual(row['response_to_packed'][0], row['prompt_length'])
                    self.assertEqual(row['ratios'][0],
                                     row['native_signed_packed'][row['prompt_length']].float())
                    self.assertEqual(row['target_offsets'], [1, 2])
                    torch.testing.assert_close(row['case']['target_ids'], row['selected'][row['prompt_length']:])


if __name__ == '__main__':
    unittest.main()
