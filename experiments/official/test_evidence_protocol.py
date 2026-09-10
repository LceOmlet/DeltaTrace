import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from evidence_protocol import (NIAH_INSTRUCTION, VT_INSTRUCTION, HP_INSTRUCTION, HP_DOCUMENTS,
    SUPPORTED_TASKS, configure_run, source_span, select_source_tokens, recovery_curve,
    sentence_recovery_curve, reference_token_ids)
from evaluate import arguments
from summarize import summarize

HERE = Path(__file__).resolve().parent


def niah_prompt():
    return NIAH_INSTRUCTION + '\nEvidence 123.\nWhat are all the special magic numbers for key? The numbers are'


def char_offsets(prompt):
    return [(i, i + 1) for i in range(len(prompt) + 1)]


class SourceProtocolTest(unittest.TestCase):
    def setUp(self):
        self.release = json.loads((HERE / 'protocol.json').read_bytes())

    def args(self, **kwargs):
        result = dict(evaluation_protocol='source-v2', sentence_recovery=False, datasets=None, ft=None,
                      selection='paper', family='qwen3')
        result.update(kwargs)
        return SimpleNamespace(**result)

    def test_new_default_has_all_evidence_tasks_and_live_ft(self):
        args = self.args()
        configure_run(args, self.release)
        self.assertEqual(args.datasets, list(SUPPORTED_TASKS))
        self.assertEqual(args.ft, 'live')
        parsed = arguments(['--family', 'qwen3', '--environment', 'unused.json', '--output', 'unused', '--selection', 'smoke'])
        self.assertEqual(parsed.evaluation_protocol, 'source-v2')

    def test_rejects_old_reference_numbers_and_unsupported_scope(self):
        for kwargs in [{'ft': 'published'}, {'datasets': ['math']}, {'selection': 'development16'}, {'datasets': []}]:
            with self.assertRaises(ValueError):
                configure_run(self.args(**kwargs), self.release)

    def test_explicit_legacy_defaults_remain_original(self):
        args = self.args(evaluation_protocol='released-v1')
        configure_run(args, self.release)
        self.assertEqual(len(args.datasets), 13)
        self.assertEqual(args.ft, 'published')
        args = self.args(evaluation_protocol='released-v1', selection='development16')
        configure_run(args, self.release)
        self.assertEqual(args.datasets, ['niah_mq_q2', 'morehopqa'])
        self.assertEqual(args.ft, 'live')

    def test_paired_reference_audit_is_only_available_with_matched_source_scope(self):
        with self.assertRaisesRegex(ValueError, 'Paired reference'):
            configure_run(self.args(evaluation_protocol='released-v1', paired_reference_audit=True), self.release)

    def test_niah_excludes_instruction_query_and_prefix(self):
        prompt = niah_prompt()
        span = source_span('niah_mq_q8', prompt)
        self.assertEqual(prompt[span['start']:span['end']], 'Evidence 123.')
        with self.assertRaises(ValueError):
            source_span('niah_mq_q8', prompt.replace('What are all', 'Where are all'))

    def test_vt_excludes_solved_example_and_current_question(self):
        prompt = VT_INSTRUCTION + '\nVAR DEMO = 1\nQuestion: demo? Answer: DEMO\n\n' + VT_INSTRUCTION + '\nVAR REAL = 2\nQuestion: current? Answer:'
        span = source_span('vt_h4_c1', prompt)
        self.assertEqual(prompt[span['start']:span['end']].strip(), 'VAR REAL = 2')
        with self.assertRaises(ValueError):
            source_span('vt_h4_c1', prompt.replace(VT_INSTRUCTION, '', 1))

    def test_hotpot_preserves_both_instructions_and_question(self):
        prompt = HP_INSTRUCTION + '\n\n' + HP_DOCUMENTS + '\n\nDocument 1:\nA fact.\n\n' + HP_INSTRUCTION + '\n\nQuestion: who? Answer:'
        span = source_span('hotpotqa_long', prompt)
        self.assertEqual(prompt[span['start']:span['end']].strip(), 'Document 1:\nA fact.')

    def test_body_leading_space_stays_with_first_evidence_token(self):
        prompt = niah_prompt().replace('\nEvidence', '\n Evidence')
        span = source_span('niah_mq_q2', prompt)
        first = span['start'] + 1
        offsets = [(0, 1), (first, first + len(' Evidence'))]
        self.assertEqual(select_source_tokens(span, offsets, [0, 1], [1]), [1])

    def test_token_scope_uses_full_containment_and_cannot_drop_gold(self):
        span = {'start': 2, 'end': 7}
        # Shift by one for the tokenizer's leading space. Tokens crossing either
        # source boundary cannot be changed safely and must remain preserved.
        offsets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        self.assertEqual(select_source_tokens(span, offsets, range(5), [2]), [2, 3])
        with self.assertRaisesRegex(ValueError, 'discard'):
            select_source_tokens(span, offsets, range(5), [1, 2])

    def test_reference_preserves_prompt_context_and_fixed_target(self):
        ids = list(range(12))
        reference = reference_token_ids(ids, [3, 4, 6], 999)
        self.assertEqual(reference, [0, 1, 2, 999, 999, 5, 999, 7, 8, 9, 10, 11])
        self.assertEqual(ids, list(range(12)))
        with self.assertRaises(ValueError):
            reference_token_ids(ids, [-1], 999)

    def test_budget_is_recomputed_after_source_filtering(self):
        scores = np.arange(400, dtype=np.float32)
        curve = recovery_curve(scores, range(100), [99], [.05, .1, .2, .3, .5])
        self.assertEqual([p['budget'] for p in curve['points']], [5, 10, 20, 30, 50])
        self.assertEqual(curve['points'][1]['precision'], .1)
        self.assertEqual(curve['points'][1]['recall'], 1.)

    def test_zero_scores_tie_deterministically_and_chance_degeneracy_is_explicit(self):
        curve = recovery_curve(np.zeros(10), range(10), [9], [.1])
        self.assertEqual(curve['points'][0]['recall'], 0)
        self.assertEqual(curve['points'][0]['recall_tie_high'], 1)
        all_gold = recovery_curve(np.ones(10), range(10), range(10), [.1])
        self.assertIsNone(all_gold['points'][0]['chance_adjusted_recall'])

    def test_sentence_metric_selects_whole_units_and_reports_token_cost(self):
        prompt = 'a b. c d.'
        # Offsets include one leading space, as in the real token map.
        offsets = [(1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (9, 10)]
        result = sentence_recovery_curve(prompt, {'start': 0, 'end': len(prompt)}, offsets,
            [6, 0, 0, 4, 4, 0], [0, 1, 3, 4], [1], [.5])
        self.assertEqual(result['budget_unit'], 'sentence_or_line_units')
        self.assertEqual(result['points'][0]['budget'], 1)
        self.assertEqual(result['points'][0]['selected_token_count'], 2)
        self.assertEqual(result['points'][0]['recall'], 0)


class SummaryProtocolTest(unittest.TestCase):
    def report(self):
        source_raw = (HERE / 'source_protocol.json').read_bytes()
        settings = json.loads(source_raw)
        curve = recovery_curve([3., 2., 1., 0.], [1, 2], [1], settings['budget_fractions'])
        metric = {'rise': .2, 'mas': .3, 'needle': 1., 'recovery': curve,
                  'reference_matches_final_deletion': True, 'actual_input_hashes': ['input', 'partial', 'reference']}
        return {'status': 'complete', 'family': 'qwen3', 'selection': 'smoke', 'selected_counts': {'niah_mq_q2': 1},
            'sample_batch': 1, 'costs': [], 'evaluation_protocol': 'source-v2', 'ft_source': 'live',
            'evaluation_settings': settings, 'evaluation_protocol_sha256': hashlib.sha256(source_raw).hexdigest(),
            'protocol_sha256': 'fixture', 'published_number_comparison': False, 'cases': [{
                'dataset': 'niah_mq_q2', 'index': 0, 'status': 'complete', 'evaluation_protocol': 'source-v2',
                'keep': [1, 2], 'author_keep': [0, 1, 2, 3], 'gold': [1], 'user_positions': [10, 11, 12, 13],
                'input_sha256': 'input',
                'source_span': {'start': 1, 'end': 4}, 'source_eligible_count': 2, 'source_gold_count': 1,
                'source_eligible_positions': [11, 12], 'reference_input_sha256': 'reference',
                'metrics': {'DT': copy.deepcopy(metric), 'FT_K1': copy.deepcopy(metric)},
                'FT_K3_needle': 1., 'FT_K3_recovery': copy.deepcopy(curve)}]}

    def test_new_summary_exposes_curves_and_never_attaches_old_ft(self):
        summary = summarize(self.report(), {})
        self.assertFalse(summary['published_number_comparison'])
        self.assertNotIn('published_FT', summary['tasks'][0])
        self.assertEqual(summary['tasks'][0]['recovery']['DT']['points'][1]['budget'], 1)

    def test_summary_rejects_protocol_mixing_and_old_ft(self):
        for change in ['case_protocol', 'published', 'old_ft', 'reference', 'budget_unit', 'budget',
                       'other_budget', 'other_eligible', 'dropped_gold', 'score_view', 'steps', 'spec']:
            r = self.report()
            if change == 'case_protocol': r['cases'][0]['evaluation_protocol'] = 'released-v1'
            if change == 'published': r['published_number_comparison'] = True
            if change == 'old_ft': r['ft_source'] = 'published'
            if change == 'reference': r['cases'][0]['metrics']['DT']['actual_input_hashes'][-1] = 'wrong'
            if change == 'budget_unit': r['cases'][0]['FT_K3_recovery']['budget_unit'] = 'sentence_or_line_units'
            if change == 'budget': r['cases'][0]['FT_K3_recovery']['points'][1]['budget'] = 40
            if change == 'other_budget': r['cases'][0]['FT_K3_recovery']['points'][4]['budget'] = 40
            if change == 'other_eligible': r['cases'][0]['FT_K3_recovery']['points'][0]['eligible'] = 4
            if change == 'dropped_gold': r['cases'][0]['gold'].append(0)
            if change == 'score_view': r['cases'][0]['FT_K3_recovery']['score_view'] = 'signed'
            if change == 'steps': r['cases'][0]['metrics']['DT']['actual_input_hashes'].pop(1)
            if change == 'spec': r['evaluation_settings']['budget_fractions'] = [.1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(r, {})

    def test_source_paper_rejects_partial_task_and_missing_weights(self):
        r = self.report()
        r['selection'] = 'paper'
        protocol = json.loads((HERE / 'protocol.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'weight'):
            summarize(r, protocol)
        r['weight_identity'] = {'verified': True}
        with self.assertRaisesRegex(ValueError, 'full released cache'):
            summarize(r, protocol)

    def test_paired_reference_summary_requires_matching_deletion_endpoint(self):
        r = self.report()
        r['paired_reference_audit'] = True
        r['cases'][0]['metrics']['DT_full_reference'] = copy.deepcopy(r['cases'][0]['metrics']['DT'])
        self.assertIn('DT_full_reference', summarize(r, {})['tasks'][0]['recovery'])
        r['cases'][0]['metrics']['DT_full_reference']['actual_input_hashes'][-1] = 'full_prompt_endpoint'
        with self.assertRaises(ValueError):
            summarize(r, {})

    def test_optional_sentence_summary_keeps_its_own_budget_and_token_cost(self):
        r = self.report()
        r['sentence_recovery_enabled'] = True
        curve = sentence_recovery_curve('a b. c d.', {'start': 0, 'end': 9},
            [(1, 2), (3, 4), (6, 7), (8, 9)], [1, 2, 3, 4], range(4), [0],
            r['evaluation_settings']['budget_fractions'])
        for method in ('DT', 'FT_K1'):
            r['cases'][0]['metrics'][method]['sentence_recovery'] = copy.deepcopy(curve)
        r['cases'][0]['FT_K3_sentence_recovery'] = copy.deepcopy(curve)
        table = summarize(r, {})['tasks'][0]
        self.assertEqual(table['sentence_recovery']['DT']['budget_unit'], 'sentence_or_line_units')
        self.assertEqual(table['sentence_recovery']['DT']['points'][1]['selected_token_count'], 2)
        self.assertEqual(table['recovery']['DT']['budget_unit'], 'source_eligible_tokens')

    def test_legacy_report_without_new_fields_still_works(self):
        report = {'status': 'complete', 'family': 'qwen3', 'selection': 'smoke', 'sample_batch': 1,
            'protocol_sha256': 'legacy', 'selected_counts': {'niah_mq_q2': 1}, 'costs': [], 'cases': [{
                'dataset': 'niah_mq_q2', 'index': 0, 'status': 'complete',
                'metrics': {'DT': {'rise': .2, 'mas': .3, 'needle': .4}}}]}
        result = summarize(report, {})
        self.assertEqual(result['evaluation_protocol'], 'released-v1')
        self.assertEqual(result['tasks'][0]['DT'], {'rise': .2, 'mas': .3, 'needle': .4})


if __name__ == '__main__':
    unittest.main()
