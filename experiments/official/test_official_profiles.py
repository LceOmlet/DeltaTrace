"""CPU checks for profile dispatch and signed-RISE/positive-MAS separation."""
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deltatrace.profiles.official import make_qwen35_runner
from evaluate import arguments, use_signed_rise


class Runner:
    def __init__(self, model, fa, fla, **options):
        self.norm_gate_rules=options.get('norm_gate_rules', {})
        self.finite_fla_by_layer=options.get('finite_fla_by_layer', {})
        self.attention_pv_rules=options.get('attention_pv_rules', {})
        self.key_norm_by_layer=options.get('key_norm_by_layer', {})


class OfficialProfiles(unittest.TestCase):
    def setUp(self):
        layers=[types.SimpleNamespace(block_type='full_attention' if i%4==3 else 'linear_attention') for i in range(32)]
        self.model=types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=layers)))
        self.backend=types.SimpleNamespace(Qwen35DenseFiniteRunner=Runner)

    def test_default_covers_every_memory_layer(self):
        with patch.dict(sys.modules, qwen35_dense_finite_runner=self.backend):
            r=make_qwen35_runner(self.model, None, lambda *args: None)
        expected={i for i in range(32) if i%4!=3}
        self.assertEqual(set(r.norm_gate_rules), expected)
        self.assertEqual(set(r.finite_fla_by_layer), expected)
        self.assertEqual(len(expected), 24)
        self.assertEqual(r.attention_pv_rules, {})
        self.assertEqual(r.key_norm_by_layer, {})

    def test_legacy_and_bad_overrides(self):
        legacy=types.SimpleNamespace(make_qwen35_clean_runner=lambda *args:Runner(*args))
        with patch.dict(sys.modules, qwen35_clean_runner=legacy, qwen35_dense_finite_runner=self.backend):
            self.assertEqual(make_qwen35_runner(self.model,None,None,profile='clean-v1').norm_gate_rules,{})
            with self.assertRaises(ValueError):make_qwen35_runner(self.model,None,lambda *args:None,norm_gate_rules={0:'symmetric'})
            with self.assertRaises(ValueError):make_qwen35_runner(self.model,None,None,profile='unknown')

    def test_cli_default_and_legacy_are_explicit(self):
        args=['--family','qwen35','--environment','env.json','--output','out','--selection','smoke']
        self.assertEqual(arguments(args).qwen35_profile,'gdn-symmetric-v1')
        self.assertEqual(arguments(args+['--qwen35-profile','clean-v1']).qwen35_profile,'clean-v1')

    def test_signed_rise_preserves_positive_mas_and_recovery(self):
        metrics={'DT':dict(rise=.3,mas=.4,needle=.8,scores=[8,3,1],actual_input_hashes=['input','p','reference']),
                 'signed':dict(rise=.2,mas=.9,needle=.1,scores=[8,2,1],actual_input_hashes=['input','s','reference'])}
        use_signed_rise(metrics,'DT','signed')
        self.assertEqual(metrics['DT']['rise'],.2)
        self.assertEqual(metrics['DT']['rise_positive'],.3)
        self.assertEqual(metrics['DT']['mas'],.4)
        self.assertEqual(metrics['DT']['needle'],.8)
        self.assertEqual(metrics['DT']['rise_curve_method'],'signed')
        bad=copy.deepcopy(metrics)
        bad['signed']['actual_input_hashes'][-1]='different reference'
        with self.assertRaises(AssertionError):use_signed_rise(bad,'DT','signed')


if __name__=='__main__':unittest.main()
