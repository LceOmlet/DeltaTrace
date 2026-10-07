"""Frozen owner boundary/transport tests; no model/DT numeric acceptance.

This local CPU has torch, but no VERL/tensordict/transformers. Tests requiring
those real dependencies are skipped; no dependency substitute is installed.
Positive native capture/cache and DT numerical accuracy are untested.
"""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
EXPECTED = {
    'reward_readout.py': '7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27',
    'deltatrace_rollout.py': '494b53b0f40831f379a427b9e47df879487e55d8bbddb4541ad0be6f3a47cace',
    'native_prefix_leases.py': 'b94756147cc6f8e59fb39baa1c2737aa655b0d9c6b87091969a65c9071ad0852',
    'qwen35_native_prefix_artifacts.py': '37a86074f430efd837c878b5409ce22ff3aac5ebdc984936ea5be5647d7b97d4',
    'qwen35_answer_finite.py': '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e',
    'counterfactual.py': '0d3412b8ac25d7c036e7a827d4bbf19a4eb2b9984a0a0956f37e82d7d86f9d07',
    'qwen35_dense_finite_runner.py': '628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f',
    'deltatrace_credit.py': '8046762ae2fd149b299e29f9a331d8ae1aed665f2de895573e78e61ebc2d8497',
    'dt_training_batch.py': '1a3f39d2ae11add49b1edb9077e915bf5f28d3f7fc9da929dd00fba79413004e',
}
TEXT_EXPECTED = {
    'reward_readout.py': '31e2acfb760eb1dd118af78a5a887898357feacc1b0f1caec04f06006a3b7d07',
    'deltatrace_rollout.py': EXPECTED['deltatrace_rollout.py'],
    'qwen35_answer_finite.py': 'd47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e',
    'qwen35_dense_finite_runner.py': '5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555',
}
FACTORY_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                           for name in ('verl', 'tensordict', 'transformers'))
FACTORY_SKIP = 'Real VERL/tensordict/transformers unavailable locally; no substitute permitted'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forbidden(*args, **kwargs):
    raise AssertionError('CPU test reached an unavailable official dependency; no substitute is provided')


class BoundaryObserved(Exception):
    pass


class ModelMarker:
    execution_device = torch.device('cpu')
    lm_head = SimpleNamespace(weight=torch.empty(0))
    _conditional = SimpleNamespace(config=None)

    def __init__(self, *, stop=False):
        self.stop, self.synchronized = stop, []

    def synchronize_prefix_start(self, start):
        self.synchronized.append(start)
        if self.stop:
            raise BoundaryObserved(start)  # GPU collective/model boundary marker only.
        return start


class RunnerMarker:
    """Replace only the GPU model attribution boundary, never its CPU bridge."""
    def __init__(self, model):
        self.model = model

    def attribute(self, pair, mask, selection, **kwargs):
        signed = torch.zeros_like(pair[1::2], dtype=torch.float64)
        signed[pair[0::2] != pair[1::2]] = .123456789123
        logp0 = torch.full(selection.labels.shape, -.6, dtype=torch.float64)
        logp1 = torch.full(selection.labels.shape, -.5, dtype=torch.float64)
        return signed, dict(root_effect=float((logp1-logp0).sum()),
                            target_logp0=logp0, target_logp1=logp1,
                            complete_attribution_seconds_with_diagnostics=0.)


def row(prompt=160, *, source_first=True, source_eos=False, uid='same-native-uid'):
    suffix = torch.tensor([80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 0])
    policy = torch.zeros_like(suffix, dtype=torch.bool)
    target = torch.zeros_like(policy)
    if source_first:
        policy[[0, 2, 10]] = True
        target[[2, 10]] = True
        if source_eos:
            suffix[0] = 0
    else:
        policy[[0, 5, 10]] = True
        target[[0, 10]] = True
    # Original prompt left padding and original terminal response padding.
    ids = torch.cat((torch.tensor([0]), torch.arange(1000, 1000+prompt), suffix))
    attention = torch.ones_like(ids)
    attention[0] = attention[-1] = 0
    return dict(input_ids=ids, responses=suffix, attention_mask=attention,
                policy_mask=policy, target_mask=target, dt_direct_reward=1., traj_uid=uid)


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, digest in EXPECTED.items():
            assert hashlib.sha256((HERE/'baseline'/name).read_bytes()).hexdigest() == digest
        for name, digest in TEXT_EXPECTED.items():
            assert hashlib.sha256((HERE/'baseline-textcraft'/name).read_bytes()).hexdigest() == digest
        cls.counterfactual = load('_frozen_prefix_counterfactual', HERE/'baseline/counterfactual.py')
        cls.bridge = load('_frozen_dt_credit_8046', HERE/'baseline/deltatrace_credit.py')
        with patch.dict(sys.modules, counterfactual=cls.counterfactual, deltatrace_credit=cls.bridge):
            cls.baseline = load('_frozen_joint_readout_7900', HERE/'baseline/reward_readout.py')
            cls.candidate = load('_candidate_joint_readout_prefix', HERE/'candidate/reward_readout.py')
            cls.text_baseline = load('_frozen_joint_readout_31e2', HERE/'baseline-textcraft/reward_readout.py')
            cls.text_candidate = load('_candidate_text_joint_readout_prefix', HERE/'candidate-textcraft/reward_readout.py')
        cls.owner = load('qwen35_native_prefix_artifacts', HERE/'baseline/qwen35_native_prefix_artifacts.py')
        cls.factory = load('_frozen_native_prefix_leases_b947', HERE/'baseline/native_prefix_leases.py')
        cls.imports = dict(qwen35_native_prefix_artifacts=cls.owner)
        cls.Targets, cls.pack = cls.owner_definitions('baseline')
        cls.TextTargets, cls.text_pack = cls.owner_definitions('baseline-textcraft')

    @staticmethod
    def owner_definitions(directory):
        # Execute the unchanged pure-CPU definitions from SHA-bound owner files.
        # Their GPU sibling imports are not loaded or replaced.
        namespace = dict(torch=torch, copy=copy)
        for filename, name in [('qwen35_answer_finite.py', 'PackedAnswerTargets'),
                               ('qwen35_dense_finite_runner.py', '_pack_native_cached_suffix_rows')]:
            path = HERE/directory/filename
            definition = next(node for node in ast.parse(path.read_bytes()).body
                              if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                         str(path), 'exec'), namespace)
        return namespace['PackedAnswerTargets'], staticmethod(namespace['_pack_native_cached_suffix_rows'])

    def readout(self, module, model, factory=None):
        return module.DirectActionTargetReadout(RunnerMarker(model),
            SimpleNamespace(eos_token_id=0), task='AppWorld', packed_answer_targets=self.Targets,
            minibatch_size=4, prefix_lease_factory=factory)

    def test_original_preparation_and_QVA_blocks_unchanged(self):
        for directories in [('baseline', 'candidate'), ('baseline-textcraft', 'candidate-textcraft')]:
            with self.subTest(task=directories):
                trees = [ast.parse((HERE/path/'reward_readout.py').read_bytes()) for path in directories]
                classes = [next(node for node in tree.body if isinstance(node, ast.ClassDef)
                                and node.name == 'DirectActionTargetReadout') for tree in trees]
                for name in ('_prepare_row', '_add_group_stats'):
                    methods = [next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                                    and node.name == name) for cls in classes]
                    self.assertEqual(ast.dump(methods[0], include_attributes=False), ast.dump(methods[1], include_attributes=False))
                methods = [next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                                and node.name == 'trajectories') for cls in classes]
                tails = []
                for method in methods:
                    index = next(i for i, node in enumerate(method.body) if isinstance(node, ast.Assign)
                                 and any(isinstance(target, ast.Name) and target.id == 'output' for target in node.targets))
                    tails.append(ast.dump(ast.Module(body=method.body[index:], type_ignores=[]), include_attributes=False))
                self.assertEqual(*tails)

    def test_producer_passes_same_existing_factory_only(self):
        before = ast.parse((HERE/'baseline/deltatrace_rollout.py').read_bytes())
        after = ast.parse((HERE/'candidate/deltatrace_rollout.py').read_bytes())
        calls = [node for node in ast.walk(after) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'DirectActionTargetReadout']
        self.assertEqual(len(calls), 1)
        factory = [keyword for keyword in calls[0].keywords if keyword.arg == 'prefix_lease_factory']
        self.assertEqual(len(factory), 1)
        self.assertEqual(ast.unparse(factory[0].value), "self.readout_options['prefix_lease_factory']")
        calls[0].keywords.remove(factory[0])
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))
        self.assertEqual((HERE/'baseline/deltatrace_rollout.py').read_bytes(),
                         (HERE/'baseline-textcraft/deltatrace_rollout.py').read_bytes())
        self.assertEqual((HERE/'candidate/deltatrace_rollout.py').read_bytes(),
                         (HERE/'candidate-textcraft/deltatrace_rollout.py').read_bytes())

    def test_each_original_no_factory_QVA_transport_unchanged(self):
        rows = [row(prompt, source_first=False) for prompt in (133, 128, 131, 130, 134, 129, 132, 135)]
        # A hole in the original response axis exercises the original owner mapping.
        for value in rows:
            value['attention_mask'][-5] = 0
        for before, after, targets in [(self.baseline, self.candidate, self.Targets),
                                       (self.text_baseline, self.text_candidate, self.TextTargets)]:
            with self.subTest(baseline=before.__name__):
                original = self.readout(before, ModelMarker())
                candidate = self.readout(after, ModelMarker())
                original.packed_answer_targets = candidate.packed_answer_targets = targets
                expected, actual = original.trajectories(rows), candidate.trajectories(rows)
                self.assertEqual(candidate.last_report['finite_trace_calls'], 2)
                for left, right in zip(expected, actual):
                    self.assertEqual(left.keys(), right.keys())
                    for key in left:
                        torch.testing.assert_close(left[key], right[key], rtol=0., atol=0.)

    def test_same_boundary_hunk_and_task_preparation_stays_distinct(self):
        app = ast.parse((HERE/'candidate/reward_readout.py').read_bytes())
        text = ast.parse((HERE/'candidate-textcraft/reward_readout.py').read_bytes())
        boundaries = []
        for tree in [app, text]:
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DirectActionTargetReadout')
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'trajectories')
            start = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'leases' for t in n.targets))
            boundaries.append(ast.dump(ast.Module(body=fn.body[start:start+2], type_ignores=[]), include_attributes=False))
        self.assertEqual(*boundaries)
        value = row(128)
        value['target_mask'][10] = False  # Real observation suffix remains after the joint target.
        app_item = self.candidate.DirectActionTargetReadout._prepare_row(value, 0)
        text_item = self.text_candidate.DirectActionTargetReadout._prepare_row(value, 0)
        self.assertLess(app_item['selected'].numel(), text_item['selected'].numel())

    def test_frozen_owner_row_suffix_pack_retains_original_IDs_targets_positions(self):
        values = [row(prompt, source_first=False, uid=str(i))
                  for i,prompt in enumerate((128, 192, 160, 224))]
        starts = (64, 128, 128, 192)
        for module, targets_type, pack in [(self.candidate, self.Targets, self.pack),
                                           (self.text_candidate, self.TextTargets, self.text_pack)]:
            with self.subTest(candidate=module.__name__):
                prepared = [module.DirectActionTargetReadout._prepare_row(v,i) for i,v in enumerate(values)]
                lengths = tuple(item['selected'].numel() for item in prepared)
                targets = targets_type([item['case'] for item in prepared],
                                       [item['target_offsets'] for item in prepared], max(lengths), 'cpu')
                paired = torch.zeros((8,max(lengths)),dtype=torch.long)
                for i,item in enumerate(prepared):
                    paired[2*i:2*i+2,:lengths[i]] = item['selected']
                provider = SimpleNamespace(prefix_lengths=starts, context_lengths=lengths, prefix_length=max(starts))
                ids,masks,positions,rebased = pack(paired,targets,provider)
                self.assertTrue(torch.equal(rebased.labels,targets.labels))
                for i,(start,length) in enumerate(zip(starts,lengths)):
                    count = length-start
                    self.assertTrue(torch.equal(ids[2*i:2*i+2,:count],paired[2*i:2*i+2,start:length]))
                    self.assertTrue(torch.equal(positions[2*i,:count],torch.arange(start,length)))
                    self.assertEqual(int(masks['linear_attention'][2*i].sum()),count)
                    self.assertEqual(int(masks['full_attention'][2*i].sum()),length)
                # The carrier192 may exceed row0's predictor127; its REAL row cut64 does not.
                self.assertGreater(provider.prefix_length, 127)
                self.assertLess(starts[0], 127)

    @unittest.skipUnless(FACTORY_DEPENDENCIES, FACTORY_SKIP)
    def test_real_factory_first_changed_boundary_and_original_floor(self):
        model = ModelMarker(stop=True)
        dt = self.readout(self.candidate, model, self.factory.prepare_native_prefix_leases)
        with patch.dict(sys.modules, self.imports), self.assertRaises(BoundaryObserved):
            dt.trajectories([row(160, uid=str(index)) for index in range(4)])
        self.assertEqual(model.synchronized, [128])  # Original b947 factory floors 160 by64.

    @unittest.skipUnless(FACTORY_DEPENDENCIES, FACTORY_SKIP)
    def test_real_factory_earliest_joint_predictor_precedes_later_source(self):
        model = ModelMarker(stop=True)
        dt = self.readout(self.candidate, model, self.factory.prepare_native_prefix_leases)
        with patch.dict(sys.modules, self.imports), self.assertRaises(BoundaryObserved):
            dt.trajectories([row(128, source_first=False, uid=str(index)) for index in range(4)])
        self.assertEqual(model.synchronized, [64])  # Predictor127, source133; old prompt128 would be invalid.

    @unittest.skipUnless(FACTORY_DEPENDENCIES, FACTORY_SKIP)
    def test_real_factory_actual_EOS_equality_does_not_invent_a_change(self):
        model = ModelMarker(stop=True)
        dt = self.readout(self.candidate, model, self.factory.prepare_native_prefix_leases)
        with patch.dict(sys.modules, self.imports), self.assertRaises(BoundaryObserved):
            dt.trajectories([row(127, source_eos=True, uid=str(index)) for index in range(4)])
        self.assertEqual(model.synchronized, [128])  # EOS-equal source127 is unchanged; earliest predictor128.

    @unittest.skipUnless(FACTORY_DEPENDENCIES, FACTORY_SKIP)
    def test_real_factory_zero_cut_and_two_B4_QVA_transport_equal(self):
        rows = [row(prompt, source_first=False) for prompt in (33, 28, 31, 30, 34, 29, 32, 35)]
        original = self.readout(self.baseline, ModelMarker()).trajectories(rows)
        model = ModelMarker()
        dt = self.readout(self.candidate, model, self.factory.prepare_native_prefix_leases)
        with patch.dict(sys.modules, self.imports), patch.object(torch.cuda, 'synchronize', return_value=None):
            actual = dt.trajectories(rows)
        self.assertEqual(model.synchronized, [0, 0])
        self.assertEqual(dt.last_report['finite_trace_calls'], 2)
        self.assertEqual(dt.last_report['shared_native_prefix']['capture_rounds'], 0)
        self.assertFalse(dt.last_report['prefix_lease_used'])
        for before, after in zip(original, actual):
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0., atol=0.)

    def test_no_requests_does_not_capture_or_collect(self):
        rows = [row() for _ in range(4)]
        for value in rows:
            value['dt_direct_reward'] = 0.
        model = ModelMarker(stop=True)
        dt = self.readout(self.candidate, model, forbidden)
        dt.trajectories(rows)
        self.assertEqual(model.synchronized, [])
        self.assertEqual(dt.last_report['finite_trace_calls'], 0)

    def test_original_target_owner_rejects_cut_past_early_predictor(self):
        for module, target_type in [(self.candidate, self.Targets), (self.text_candidate, self.TextTargets)]:
            with self.subTest(candidate=module.__name__):
                prepared = [module.DirectActionTargetReadout._prepare_row(row(128, source_first=False), i)
                            for i in range(4)]
                targets = target_type([item['case'] for item in prepared],
                                      [item['target_offsets'] for item in prepared],
                                      prepared[0]['selected'].numel(), 'cpu')
                rebased = targets.suffix(64)
                self.assertTrue(torch.equal(rebased.labels, targets.labels))
                self.assertTrue(torch.equal(rebased.positions + 64, targets.positions))
                with self.assertRaisesRegex(ValueError, 'Every target predictor'):
                    targets.suffix(128)
                rows = targets.suffix_rows([64]*4, [item['selected'].numel() for item in prepared])
                self.assertTrue(torch.equal(rows.labels, targets.labels))
                with self.assertRaisesRegex(ValueError, 'Every original target predictor'):
                    targets.suffix_rows([128]*4, [item['selected'].numel() for item in prepared])

    def test_original_lease_rejects_wrong_shape_and_exact_prefix_ID(self):
        ids = torch.arange(4*128).reshape(4, 128)
        source = self.owner.NativePrefixArtifacts(None, ids, [])
        sources = [(source, index) for index in range(4)]
        for kwargs in ({}, dict(prefix_lengths=(64, 128, 64, 128), context_lengths=(160,)*4)):
            lease = self.owner.NativePrefixLease(sources, 128, **kwargs)
            with self.assertRaisesRegex(ValueError, 'shapes differ'):
                lease(ids[:, :64])
            wrong = ids.clone()
            wrong[0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'exact factual token IDs'):
                lease(wrong)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
