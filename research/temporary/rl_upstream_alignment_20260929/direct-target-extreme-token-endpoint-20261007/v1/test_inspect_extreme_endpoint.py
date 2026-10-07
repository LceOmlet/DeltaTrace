"""CPU transport checks on the two saved real B4s, not model accuracy tests."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('extreme_endpoint', HERE/'inspect_extreme_endpoint.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

NATIVES = {
    'appworld': AUDIT/'direct-target-prefix-runtime-20261007/v1/first-native-artifacts/rank1-readout-native-batch-6.pt',
    'textcraft': AUDIT/'direct-target-prefix-runtime-20261007/v1/textcraft-credit-cpu-complete-1791373053/rank1-readout-native-batch-21.pt',
}


def definitions(path, names, expected_sha):
    """Execute only unchanged CPU definitions from the exact frozen owner."""
    if probe.sha(path) != expected_sha:
        raise AssertionError('Frozen owner source SHA changed: '+str(path))
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('Missing original definition')
    namespace = dict(torch=torch, copy=copy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class OriginalOwnerTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = {}
        for name, case in probe.CASES.items():
            path = AUDIT/f'direct-target-prefix-runtime-20261007/v1/current-runtime-metadata/{name}-source.json'
            source, native, rows = probe.load_request(path, NATIVES[name], case)
            head = (AUDIT/'direct-target-prefix-interface-20261007/v1/baseline-textcraft/qwen35_answer_finite.py'
                    if name == 'textcraft' else AUDIT/'direct-target-head-memory-20261007/v1/candidate/qwen35_answer_finite.py')
            owners = definitions(head, ['PackedAnswerTargets'],
                source['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py'])
            selector = AUDIT/'direct-target-causal-prefix-20261007/v3/memory-source/clean/qwen35/native_target_logit_rows.py'
            owners.update(definitions(selector, ['NativeTargetLogitRows'],
                source['dt_source_sha256']['clean/qwen35/native_target_logit_rows.py']))
            pad = AUDIT/'recipe-sources/verl-agent-20bd331/verl/utils/torch_functional.py'
            owners.update(definitions(pad, ['pad_2d_list_to_length'],
                source['verl_sha256']['verl/utils/torch_functional.py']))
            cls.inputs[name] = (case, source, native, rows, owners)

    def test_actual_inputs_and_source_identity(self):
        for name, (case, source, native, rows, _) in self.inputs.items():
            with self.subTest(case=name):
                result = probe.geometry(source, native, rows, case)
                self.assertEqual(result['candidate']['token_id'], case['token_id'])
                self.assertEqual(result['candidate']['packed_slot'], case['packed_slot'])
                self.assertEqual(result['candidate']['saved_native_signed_dtype'], 'torch.float64')
                self.assertEqual(result['operations']['DT'], 0)
                self.assertFalse(torch.cuda.is_initialized())

    def test_actor_only_forwards_original_resolved_scheduler_scalar(self):
        self.assertEqual(probe.actor_initialization_steps(200, None), 200)
        self.assertEqual(probe.actor_initialization_steps(None, 330), 330)
        with self.assertRaises(ValueError):
            probe.actor_initialization_steps(None, None)
        with self.assertRaises(ValueError):
            probe.actor_initialization_steps(200, 330)

    def test_only_evidenced_source_changes_and_all_original_rows_survive(self):
        for name, (case, _, native, rows, owners) in self.inputs.items():
            with self.subTest(case=name):
                pad = owners['pad_2d_list_to_length']
                # EOS identity is irrelevant to this transport test; it is a
                # distinct marker and the GPU probe reads the actual tokenizer.
                marker_eos = 248044
                pair = probe.make_pair(rows, native['native_signed'].shape[1], marker_eos, pad, case)
                original = pad([row['selected'].tolist() for row in rows], marker_eos,
                    max_length=native['native_signed'].shape[1])
                self.assertTrue(torch.equal(pair[1::2], original))
                changes = pair[0::2].ne(original).nonzero().tolist()
                self.assertEqual(changes, [[case['row'], case['packed_slot']]])
                self.assertEqual(int(pair[2*case['row'], case['packed_slot']]), marker_eos)
                self.assertEqual(tuple(pair.shape), (8, native['native_signed'].shape[1]))

    def test_original_joint_target_selection_and_packing(self):
        for name, (_, _, native, rows, owners) in self.inputs.items():
            with self.subTest(case=name):
                selection = owners['PackedAnswerTargets']([r['case'] for r in rows],
                    [r['target_offsets'] for r in rows], native['native_signed'].shape[1], 'cpu')
                selector = owners['NativeTargetLogitRows'](selection)
                # A one-column row marker checks transport, never vocabulary scores.
                marker = torch.arange(8*selector.rows.numel()).view(8, -1, 1)
                actual = selector.pack_logits(marker).flatten()
                expected = marker[selection.paired_samples, selector.packed_rows, 0]
                self.assertTrue(torch.equal(actual, expected))
                at = 0
                for i, row in enumerate(rows):
                    count = len(row['target_offsets'])
                    self.assertEqual(selection.samples[at:at+count].tolist(), [i]*count)
                    self.assertEqual(selection.positions[at:at+count].tolist(),
                        [row['prompt_length']+offset-1 for offset in row['target_offsets']])
                    self.assertTrue(torch.equal(selection.labels[at:at+count],
                        row['case']['target_ids'][row['target_offsets']]))
                    at += count

    def test_snapshot_preserves_fp32_selection_and_original_fp64_sum_interface(self):
        for name, (case, _, native, rows, owners) in self.inputs.items():
            with self.subTest(case=name):
                selection = owners['PackedAnswerTargets']([r['case'] for r in rows],
                    [r['target_offsets'] for r in rows], native['native_signed'].shape[1], 'cpu')
                marker_logp = -torch.arange(len(selection.paired_samples), dtype=torch.float32)/10000
                saved = probe.target_snapshot(selection, marker_logp)
                self.assertTrue(torch.equal(saved['reference_target_logp'], marker_logp[0::2]))
                self.assertTrue(torch.equal(saved['factual_target_logp'], marker_logp[1::2]))
                self.assertEqual(saved['factual_target_logp'].dtype, torch.float32)
                self.assertTrue(torch.equal(saved['predictor_positions'], selection.positions))
                self.assertTrue(torch.equal(saved['labels'], selection.labels))
                joint = selection.sample_sums(saved['factual_target_logp'].double())
                self.assertEqual(joint.dtype, torch.float64)
                self.assertEqual(joint.shape, (4,))
                description = probe.causal_description(saved,
                    native['native_signed'][case['row'], case['packed_slot']], case)
                self.assertEqual(description['earlier_target_count']+description['future_target_count'],
                    len(rows[case['row']]['target_offsets']))
                self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
