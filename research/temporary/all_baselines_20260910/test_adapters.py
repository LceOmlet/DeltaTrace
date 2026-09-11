import unittest
import numpy as np
from baseline_adapters import aggregate_matrix,validate_target

class AdapterTests(unittest.TestCase):
    def test_negative_values_survive_before_sentence_sum(self):
        raw=np.array([[10.,-9,0],[2,1,np.nan],[np.nan]*3],np.float32)
        signed,native=aggregate_matrix(raw,[1,1,0],2)
        np.testing.assert_array_equal(signed,[12,-8,0]);self.assertGreater(native[1],0)
    def test_whole_response_weights_include_reasoning(self):
        tokens=['Reasoning',',','\n',' answer','.','<eos>']
        self.assertEqual(validate_target([1,0,0,1,0,0],tokens),[1,0,0,1,0,0])
        with self.assertRaises(ValueError):validate_target([0,0,0,1,0,0],tokens)
    def test_missing_selected_prompt_rows_fail(self):
        with self.assertRaises(ValueError):aggregate_matrix([[1,np.nan],[2,3]],[1,1],2)
        with self.assertRaises(ValueError):aggregate_matrix([[1,2]],[0],2)
    def test_unselected_eos_nan_is_allowed(self):
        signed,_=aggregate_matrix([[1,-2],[np.nan,np.nan]],[1,0],2)
        np.testing.assert_array_equal(signed,[1,-2])
    def test_native_diagnostic_does_not_change_signed(self):
        raw=np.array([[2.,-1],[1.,3]],np.float32);saved=raw.copy()
        signed,native=aggregate_matrix(raw,[1,1],2)
        np.testing.assert_array_equal(raw,saved);np.testing.assert_array_equal(signed,[3,2])
        np.testing.assert_allclose(native,[1.25,.75])

if __name__=='__main__':unittest.main()
