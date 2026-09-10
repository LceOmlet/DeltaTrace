"""Tests for preserved conditioning, full costs and nested retrieval."""
import unittest
from artifacts_v3 import ROOT
from hotpot_retrieval_v3 import answer_conditioned_weights,all_token_groups,rank_sentences,select_prefix
from hotpot_evidence import native_units

class ProtocolTests(unittest.TestCase):
    def test_seed_mask_does_not_reconstruct_input(self):
        tokens=['Reasoning',' says',' yes','.','\n','Yes','.','<eos>'];saved=list(tokens)
        self.assertEqual(answer_conditioned_weights(tokens,(5,6),7),[0,0,0,0,0,1,0,0])
        self.assertEqual(tokens,saved)

    def test_invalid_answer_target_fails(self):
        with self.assertRaises(ValueError):answer_conditioned_weights(['x','<eos>'],(0,1),1)
        with self.assertRaises(ValueError):answer_conditioned_weights([' ','.','<eos>'],(0,1),2)

    def test_full_cost_counts_punctuation_and_space(self):
        text='T\nA. B.';units=native_units(text,[['T',['A.',' B.']]],0,len(text))
        offsets=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]
        groups=all_token_groups(text,offsets,units,coordinate_shift=0)
        self.assertEqual(groups,[ [0,1],[2,3],[4,5,6] ])
        self.assertEqual(select_prefix([1,2],groups,token_budget=4),([1],[2,3]))
        self.assertEqual(len(select_prefix([1,2],groups,token_budget=5)[1]),5)

    def test_old_skip_counterexample_is_nested(self):
        groups=[list(range(6)),[6,7,8],[9,10]]
        self.assertEqual(select_prefix([0,1,2],groups,token_budget=5),([],[]))
        self.assertEqual(select_prefix([0,1,2],groups,token_budget=6),([0],list(range(6))))
        previous=set()
        for budget in range(12):
            ids,tokens=select_prefix([0,1,2],groups,token_budget=budget)
            self.assertTrue(previous<=set(ids));self.assertLessEqual(len(tokens),budget);previous=set(ids)

    def test_top_sentences_charge_same_full_tokens(self):
        ids,tokens=select_prefix([0,1],[[0,1,2],[3,4]],sentence_budget=2)
        self.assertEqual(ids,[0,1]);self.assertEqual(len(tokens),5)
        with self.assertRaises(ValueError):select_prefix([0],[[0]],token_budget=1,sentence_budget=1)

    def test_signed_sum_preserves_cancellation(self):
        units=[dict(kind='sentence',start=0),dict(kind='sentence',start=5)]
        order,pooled=rank_sentences([10,-9,2],units,[[0,1],[2]],[[0,1],[2]],pooling='signed_sum')
        self.assertEqual(order,[1,0]);self.assertEqual(pooled,{0:1.,1:2.})
        order,_=rank_sentences([10,-9,2],units,[[0,1],[2]],[[0,1],[2]],pooling='positive_mean_eligible')
        self.assertEqual(order,[0,1])

    def test_no_metadata_candidate_or_gold_argument(self):
        import inspect
        u=[dict(kind='metadata',start=0),dict(kind='sentence',start=1),dict(kind='sentence',start=2)]
        order,_=rank_sentences([99,0,0],u,[[0],[1],[2]],[[0],[1],[2]],pooling='signed_sum')
        self.assertEqual(order,[1,2])
        self.assertTrue(set(inspect.signature(rank_sentences).parameters).isdisjoint({'gold','labels','target','answer'}))

    def test_costs_cannot_duplicate_tokens(self):
        with self.assertRaises(ValueError):select_prefix([0,1],[[0],[0]],token_budget=2)
        with self.assertRaises(ValueError):select_prefix([0],[[0]],token_budget=1.5)

if __name__=='__main__':unittest.main()
