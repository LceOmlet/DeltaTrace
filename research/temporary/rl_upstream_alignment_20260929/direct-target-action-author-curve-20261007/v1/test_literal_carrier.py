"""Actual saved B4 ID transport only; the GPU author curve is separate evidence."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

HERE=Path(__file__).resolve().parent
ENDPOINT=HERE.parents[1]/'direct-target-extreme-token-endpoint-20261007/v1'
sys.path.insert(0,str(ENDPOINT))
import test_inspect_extreme_endpoint as owner_test
spec=importlib.util.spec_from_file_location('action_curve',HERE/'inspect_action_curve.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class ActualLiteralCarrier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owner_test.OriginalOwnerTransport.setUpClass()
        cls.case,cls.source,cls.native,cls.rows,cls.owners=owner_test.OriginalOwnerTransport.inputs['appworld']
        cls.tokenizer=SimpleNamespace(eos_token_id=248044,eos_token='<|im_end|>')

    def test_original_ids_concatenate_without_extra_EOS(self):
        row=self.rows[0];ids=probe.LiteralIds(row,self.tokenizer)
        prefix=ids(probe.FORMATTED,return_tensors='pt',add_special_tokens=False).input_ids
        suffix=ids(probe.GENERATION+ids.eos_token,return_tensors='pt',add_special_tokens=False).input_ids
        self.assertTrue(torch.equal(torch.cat((prefix,suffix),dim=1)[0],row['selected']))
        self.assertEqual(suffix.numel(),1)
        self.assertEqual(int(suffix[0,0]),int(row['case']['target_ids'][row['target_offsets'][-1]]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_source_positions_are_disjoint_from_original_target(self):
        row=self.rows[0]
        positions=(row['prior'][row['suffix_positions']].nonzero().flatten()+row['prompt_length'])
        targets=row['prompt_length']+torch.tensor(row['target_offsets'])
        self.assertFalse(bool(torch.isin(positions,targets).any()))
        self.assertTrue(bool((positions<row['selected'].numel()-1).all()))
        self.assertEqual(positions.numel(),1266)
        self.assertTrue(bool(positions.eq(self.case['packed_slot']).any()))


if __name__=='__main__':unittest.main(verbosity=2)
