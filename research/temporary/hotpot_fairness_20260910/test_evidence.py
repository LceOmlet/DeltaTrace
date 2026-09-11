"""Regression tests for concrete label, boundary, and retrieval-cost failures."""
import inspect
import unittest
from common import ROOT
from hotpot_evidence import (native_units,token_groups,sentence_order,select_sentences,
                            supporting_fact_metrics,fact_recall_ceiling)

class EvidenceTests(unittest.TestCase):
    def test_duplicate_title_maps_to_body(self):
        text='Document 1:\nWhy Is There Air?\nWhy Is There Air? Extra.'
        units=native_units(text,[['Why Is There Air?',['Why Is There Air?',' Extra.']]],0,len(text))
        fact=next(u for u in units if u['kind']=='sentence')
        self.assertEqual(fact['start'],text.rfind('Why Is There Air?'))
        self.assertNotEqual(fact['start'],text.find('Why Is There Air?'))

    def test_leading_space_belongs_to_next_sentence(self):
        text='T\nAlpha. Beta.';units=native_units(text,[['T',['Alpha.',' Beta.']]],0,len(text))
        groups=token_groups(text,[(2,7),(7,8),(8,13),(13,14)],[0,1,2,3],units,coordinate_shift=0)
        self.assertEqual(groups,[[],[0,1],[2,3]])

    def test_crossing_token_charged_once(self):
        text='T\n.ky';units=native_units(text,[['T',['.','ky']]],0,len(text))
        groups=token_groups(text,[(2,4),(4,5)],[0,1],units,coordinate_shift=0)
        self.assertEqual(groups,[[],[0],[1]])

    def test_native_ids_preserved(self):
        text='T\nA.B';units=native_units(text,[['T',['A.','','B']]],0,len(text))
        self.assertEqual([u['sentence_index'] for u in units if u['kind']=='sentence'],[0,2])

    def test_whole_sentence_budget_skip_no_free_expansion(self):
        groups=[list(range(8)),[8,9,10],[11,12]]
        selected,tokens=select_sentences([0,1,2],groups,token_budget=5)
        self.assertEqual(selected,[1,2]);self.assertEqual(len(tokens),5)
        self.assertEqual(select_sentences([0,1,2],groups,token_budget=2),([2],[11,12]))
        self.assertEqual(len(select_sentences([0,1,2],groups,sentence_budget=2)[1]),11)
        with self.assertRaises(ValueError):select_sentences([0,1,2],groups,token_budget=5,sentence_budget=2)

    def test_equal_fact_weight_and_exact_set_metric(self):
        units=[dict(kind='sentence',title='T',sentence_index=i,start=i) for i in range(3)]
        groups=[list(range(10)),[10],[11]];gold=[('T',0),('T',1)]
        self.assertEqual(supporting_fact_metrics([0],units,gold)['recall'],.5)
        self.assertEqual(supporting_fact_metrics([1],units,gold)['recall'],.5)
        metrics=supporting_fact_metrics([0,1,2],units,gold)
        self.assertEqual(metrics['complete_support'],1);self.assertEqual(metrics['exact_match'],0)
        self.assertAlmostEqual(metrics['precision'],2/3);self.assertAlmostEqual(metrics['f1'],.8)
        self.assertEqual(fact_recall_ceiling(units,groups,gold,token_budget=10),.5)
        self.assertEqual(fact_recall_ceiling(units,groups,gold,token_budget=11),1)

    def test_rank_is_gold_free_and_excludes_metadata(self):
        units=[dict(kind='metadata',start=0),dict(kind='sentence',start=2),dict(kind='sentence',start=4)]
        self.assertEqual(sentence_order([100,0,0],units,[[0],[1],[2]]),[1,2])
        for fn in (native_units,token_groups,sentence_order):
            self.assertTrue(set(inspect.signature(fn).parameters).isdisjoint({'gold','gold_keys','answer','target'}))

    def test_invalid_source_fails_closed(self):
        with self.assertRaises(ValueError):native_units('T\nA.T\nA.',[['T',['A.']]],0,10)
        with self.assertRaises(ValueError):native_units('T\nA.',[['T',['Different.']]],0,4)

if __name__=='__main__':unittest.main()
