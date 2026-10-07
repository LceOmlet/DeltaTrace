"""Verify the diagnosis transport on the saved real B4, never model accuracy."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
ENDPOINT = HERE.parents[1] / 'direct-target-extreme-token-endpoint-20261007/v1'
sys.path.insert(0, str(ENDPOINT))
from test_inspect_extreme_endpoint import OriginalOwnerTransport

spec = importlib.util.spec_from_file_location('reference_context', HERE / 'inspect_reference_context.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ActualReferenceContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OriginalOwnerTransport.setUpClass()
        cls.case, cls.source, cls.native, cls.rows, cls.owners = OriginalOwnerTransport.inputs['appworld']
        cls.eos = 248044
        cls.width = cls.native['native_signed'].shape[1]
        pad = cls.owners['pad_2d_list_to_length']
        cls.original = pad([row['selected'].tolist() for row in cls.rows], cls.eos, max_length=cls.width)
        cls.pair = probe.reference_context_pair(cls.rows, cls.width, cls.eos, pad, cls.case)

    def test_pair_differs_only_at_actual_extreme_token(self):
        self.assertEqual(self.pair[0::2].ne(self.pair[1::2]).nonzero().tolist(), [[0, 7260]])
        self.assertEqual(int(self.pair[0, 7260]), self.eos)
        self.assertEqual(int(self.pair[1, 7260]), 198)
        self.assertEqual(tuple(self.pair.shape), (8, 9773))

    def test_other_rows_and_original_targets_survive(self):
        self.assertTrue(torch.equal(self.pair[2::2], self.original[1:]))
        self.assertTrue(torch.equal(self.pair[3::2], self.original[1:]))
        for row_index, row in enumerate(self.rows):
            positions = row['prompt_length'] + torch.tensor(row['target_offsets'])
            for pair_index in (2*row_index, 2*row_index+1):
                self.assertTrue(torch.equal(self.pair[pair_index, positions], self.original[row_index, positions]))

    def test_changed_context_is_only_original_prior_source(self):
        changed = self.pair[0].ne(self.original[0]).nonzero().flatten()
        self.assertEqual(changed.numel(), 1266)
        row = self.rows[0]
        original_slots = row['suffix_positions'][changed - row['prompt_length']]
        self.assertTrue(bool(row['prior'][original_slots].all()))
        self.assertFalse(bool(row['target'][original_slots].any()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
