import unittest

import numpy as np

from recovery_diagnostics import recovery_diagnostics, reported_recovery_diagnostics
from retrieval_views import sentence_density_order, restrict_order_to_span


class RecoveryDiagnosticsTest(unittest.TestCase):
    def test_perfect_ranking_can_have_low_recall(self):
        # 8 evidence tokens, 2 predictions: a perfect selector only recalls 25%.
        d = recovery_diagnostics(np.arange(20, 0, -1), range(20), range(8))
        self.assertEqual(d['budget'], 2)
        self.assertEqual(d['ceiling'], .25)
        self.assertEqual(d['recall'], .25)
        self.assertEqual(d['ceiling_adjusted_recall'], 1)
        self.assertEqual(d['precision'], 1)

    def test_filter_before_denominator_and_round_up(self):
        d = recovery_diagnostics([3, 2, 1, 0], [0, 0, 2, 3, -1, 9], [0, 1, 9], .4)
        self.assertEqual((d['eligible'], d['gold'], d['budget']), (3, 1, 2))
        self.assertEqual(d['random_expected_recall'], 2 / 3)

    def test_tie_interval_covers_any_topk_choice(self):
        d = recovery_diagnostics([1] * 10, range(10), [9], .1)
        self.assertEqual(d['recall_tie_low'], 0)
        self.assertEqual(d['recall_tie_high'], 1)
        self.assertEqual(d['selected'], [0])

    def test_empty_and_nonfinite_fail_explicitly(self):
        for scores, keep, gold in [([1], [], [0]), ([1], [0], []), ([np.nan], [0], [0])]:
            with self.assertRaises(ValueError):
                recovery_diagnostics(scores, keep, gold)

    def test_report_preserves_author_ties_and_rejects_wrong_alignment(self):
        d = reported_recovery_diagnostics([1] * 10, range(10), [9], 1)
        self.assertEqual(d['reported_recall'], 1)
        self.assertEqual(d['deterministic_cpu_recall'], 0)
        self.assertEqual(d['ceiling_adjusted_recall'], 1)
        with self.assertRaises(ValueError):
            reported_recovery_diagnostics([10, 1], [0, 1], [0], 0)

    def test_pooling_uses_eligible_density_and_returns_rank_only(self):
        text = 'a b. c d.'
        offsets = [(0, 1), (2, 3), (3, 4), (5, 6), (7, 8), (8, 9)]
        scores = [6., 0., 999., 4., 4., 999.]
        order = sentence_density_order(text, offsets, scores, [0, 1, 3, 4])
        # First segment density=3, second=4; punctuation is ineligible.
        self.assertEqual(order, [3, 4, 0, 1])
        self.assertEqual(scores, [6., 0., 999., 4., 4., 999.])
        self.assertEqual(order, sentence_density_order(text, offsets, np.array(scores) * 2, [0, 1, 3, 4]))

    def test_newline_segmentation_and_absolute_view(self):
        text = 'a b\nc d'
        offsets = [(0, 1), (2, 3), (4, 5), (6, 7)]
        scores = [-8, 0, 3, 3]
        self.assertEqual(sentence_density_order(text, offsets, scores, range(4)), [2, 3, 0, 1])
        self.assertEqual(sentence_density_order(text, offsets, scores, range(4), absolute=True), [0, 1, 2, 3])

    def test_bad_offsets_rejected(self):
        with self.assertRaises(ValueError):
            sentence_density_order('a', [(0, 2)], [1], [0])

    def test_source_scope_preserves_order_and_token_overlap(self):
        self.assertEqual(restrict_order_to_span([3, 1, 0, 2], [(0, 2), (2, 5), (5, 7), (7, 9)], 4, 8), [3, 1, 2])
        with self.assertRaises(ValueError):
            restrict_order_to_span([], [], 5, 5)


if __name__ == '__main__':
    unittest.main()
