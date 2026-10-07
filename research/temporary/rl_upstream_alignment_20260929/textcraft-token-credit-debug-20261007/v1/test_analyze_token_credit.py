"""CPU schema/mapping tests using frozen prepare and v2 save interfaces.

Literal tensor markers test transport only, not DT, Q/V/A or whitening accuracy.
No model, forward, GPU, RPC or environment evaluation is called.
"""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import types
import unittest

import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = load('credit_analyzer', HERE / 'analyze_token_credit.py')
observer = load('credit_observer_v2', HERE.parent / 'v2/observe_token_credit.py')


def frozen_prepare(relative, expected):
    path = AUDIT / 'direct-target-prefix-interface-20261007/v1' / relative / 'reward_readout.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise AssertionError('test owner source SHA differs: ' + str(path))
    owner = next(node for node in ast.parse(raw).body
                 if isinstance(node, ast.ClassDef) and node.name == 'DirectActionTargetReadout')
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == '_prepare_row')
    cls = ast.ClassDef(name='FrozenPrepare', bases=[], keywords=[], body=[method], decorator_list=[])
    namespace = dict(torch=torch, math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['FrozenPrepare']._prepare_row


PREPARE_TEXT = frozen_prepare('candidate-textcraft', '814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b')
PREPARE_APP = frozen_prepare('candidate', 'b60251fdd3abd1687e5f8d6ca7415575acb0c88e60be30892a7d1513fe82c17c')


def row(uid='same', tail=25):
    # Left-padded prompt and a response attention gap distinguish all axes.
    return dict(input_ids=torch.tensor([0, 11, 12, 21, 0, 22, 23, 24, tail]),
                responses=torch.tensor([21, 0, 22, 23, 24, tail]),
                attention_mask=torch.tensor([0, 1, 1, 1, 0, 1, 1, 1, 1]),
                policy_mask=torch.tensor([1, 0, 1, 1, 1, 1], dtype=torch.bool),
                target_mask=torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.bool),
                dt_direct_reward=torch.tensor(1.), traj_uid=uid)


TRACE = dict(factual_target_logp=-7.25, reference_target_logp=-8.5, root_effect=1.25)
OUTPUT = dict(dt_token_advantages=torch.tensor([-3., 0., -8., 1., 0., 0.]),
              dt_q_estimates=torch.tensor([1., 0., 1., 1., 1., 1.]),
              dt_v_estimates=torch.tensor([4., 0., 9., 0., 1., 1.]))


def saved_row(prepare=PREPARE_TEXT, original=None, index=0):
    item = prepare(original or row(), index)
    # Explicit independent markers: neither d nor credit is recomputed here.
    item['ratios'][0], item['ratios'][2] = -1.25, -2.5
    native = torch.zeros(item['selected'].numel() + 1, dtype=torch.float64)
    native[2], native[3] = -1.234567890123, -2.345678901234
    return observer._row_record(item, OUTPUT, TRACE, native)


def save_readout(directory, rows):
    path = directory / 'rank0-readout.pt'
    torch.save(dict(scope='CPU transport markers through original v2 _row_record',
                    pid=123, captured_unix=1., provenance={'cpu_schema_test': True},
                    rows=rows, report={}, observer_errors=[]), path)


def save_actor(directory, *, tail=25, late_minimum=False, artifact=True):
    retained = [1, 2, 4]  # Original artifact response offsets -> original slots 2,3,5.
    tensors = dict(responses=torch.tensor([[22, 23, tail]]),
                   input_ids=torch.tensor([[0, 11, 22, 23, tail]]),
                   response_mask=torch.ones((1, 3), dtype=torch.bool),
                   advantages=torch.tensor([[-9., 2., -10. if late_minimum else -4.]]))
    for key, value in OUTPUT.items():
        tensors[key] = value[torch.tensor([2, 3, 5])][None, :]
    non_tensors = dict(traj_uid=['same'])
    if artifact:
        non_tensors['dt_direct_target_artifact'] = [dict(
            prompt_ids=[11, 12], response_ids=[21, 22, 23, 24, tail],
            policy_mask=[True] * 5, target_mask=[False, False, True, False, False],
            retained_response_positions=retained)]
    payload = observer._actor_snapshot(types.SimpleNamespace(batch=tensors, non_tensor_batch=non_tensors),
                                       {'cpu_schema_test': True}, directory / 'rank0-readout.pt')
    torch.save(payload, directory / 'rank0-pre-update.pt')


# The actual v2 wrapper writes native-batch files before this original call fails.
def trace_token_attribution(_runner, _reference, _factual, cases, _offsets):
    length = max(case['prompt_length'] + case['target_ids'].numel() for case in cases) + 1
    signed = torch.zeros((len(cases), length), dtype=torch.float64)
    signed[:, 2], signed[:, 3] = -1.234567890123, -2.345678901234
    detail = TRACE if len(cases) == 1 else dict(per_sample=[dict(TRACE) for _ in cases])
    return signed, None, detail


class NativeOnlyOwner:
    _prepare_row = staticmethod(PREPARE_APP)

    def trajectories(self, rows):
        items = [self._prepare_row(value, index) for index, value in enumerate(rows)]
        trace_token_attribution(None, None, None, [item['case'] for item in items],
                                [item['target_offsets'] for item in items])
        raise RuntimeError('later owner batch failed; CPU schema marker only')


def save_native_only(directory, rows):
    owner = NativeOnlyOwner()
    observation = observer.ReadoutObservation(owner, directory / 'rank0-readout.pt',
                                              provenance={'cpu_schema_test': True}).install()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'later owner batch'):
        owner.trajectories(rows)
    assert observation.restored and not observation.saved


class AnalyzeSchemaTests(unittest.TestCase):
    def analyze(self, directory):
        return analyzer.analyze(directory, [], expected_ranks=(0,))

    def test_text_complete_original_v2_schema_and_dtypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_readout(directory, [saved_row()])
            save_actor(directory)
            result = self.analyze(directory)
        self.assertEqual(result['errors'], [])
        self.assertTrue(result['all_expected_files_present'])
        entry = result['actor_extreme_links'][0]['matching_readout_extremal_slot_records'][0]
        self.assertEqual((entry['response_slot'], entry['packed_input_slot'], entry['original_input_slot']), (2, 3, 5))
        self.assertEqual(entry['native_signed_dtype'], 'torch.float64')
        self.assertEqual(entry['consumed_d_dtype'], 'torch.float32')
        self.assertEqual(entry['native_signed'], -2.345678901234)
        self.assertEqual(entry['consumed_d_storage'], -2.5)
        self.assertEqual(entry['joint_factual_target_logp'], TRACE['factual_target_logp'])
        self.assertEqual(result['actor_extreme_links'][0]['exact_token_and_raw_QVA_match_candidates'], 1)
        self_target = result['readout_statistics']['consumed_d_storage/self_target']['finite_extrema']['min']
        self.assertEqual(self_target['d_role'], 'self_target_literal_boundary')
        self.assertEqual(self_target['consumed_d_storage'], 0.)

    def test_app_trim_attention_gap_and_late_actor_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_readout(directory, [saved_row(PREPARE_APP)])
            save_actor(directory, late_minimum=True)
            result = self.analyze(directory)
        self.assertEqual(result['errors'], [])
        entry = result['actor_extreme_links'][0]['matching_readout_extremal_slot_records'][0]
        self.assertEqual((entry['response_slot'], entry['original_input_slot'], entry['attention_effective_index']), (5, 8, 6))
        self.assertIsNone(entry['packed_input_slot'])
        self.assertIsNone(entry['native_signed'])
        self.assertEqual(entry['raw_QVA']['dt_token_advantages'], 0.)
        self.assertEqual(result['actor_extreme_links'][0]['exact_token_and_raw_QVA_match_candidates'], 1)

    def test_same_uid_same_compute_prefix_different_full_tail_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_readout(directory, [saved_row(PREPARE_APP), saved_row(PREPARE_APP, row(tail=99), 1)])
            save_actor(directory)
            result = self.analyze(directory)
        self.assertEqual(result['errors'], [])
        self.assertEqual([item['saved_row'] for item in result['actor_rows'][0]['matching_readout_rows']], [0])

    def test_duplicate_exact_artifacts_remain_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_readout(directory, [saved_row(PREPARE_APP), saved_row(PREPARE_APP, index=1)])
            save_actor(directory)
            result = self.analyze(directory)
        link = result['actor_extreme_links'][0]
        self.assertTrue(link['no_candidate_selected_when_multiple'])
        self.assertEqual(link['exact_token_and_raw_QVA_match_candidates'], 2)

    def test_actual_v2_native_batch_survives_later_failure_without_qva(self):
        original_trace = trace_token_attribution
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_native_only(directory, [row(), row(uid='second')])
            result = self.analyze(directory)
        self.assertIs(trace_token_attribution, original_trace)
        self.assertEqual(result['errors'], [])
        self.assertFalse(result['all_expected_files_present'])
        self.assertEqual(result['counts']['native_batch_work_rows'], 2)
        self.assertEqual(result['readout_statistics'], {})
        self.assertEqual(result['actor_statistics'], {})
        stat = result['native_batch_statistics']['native_signed/prior_source']
        self.assertEqual(stat['tokens'], 4)
        entry = stat['finite_extrema']['min']
        self.assertEqual((entry['response_slot'], entry['packed_input_slot']), (2, 3))
        self.assertEqual(entry['raw_QVA'], {})
        self.assertIsNone(entry['consumed_d_storage'])
        self.assertEqual(entry['joint_all_prior_EOS_reference_target_logp'], TRACE['reference_target_logp'])

    def test_native_snapshots_not_added_to_completed_readout_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_native_only(directory, [row()])
            save_readout(directory, [saved_row(PREPARE_APP)])
            result = self.analyze(directory)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['readout_statistics']['native_signed/prior_source']['tokens'], 2)
        self.assertEqual(result['native_batch_statistics']['native_signed/prior_source']['tokens'], 2)

    def test_missing_artifact_keeps_actor_local_values_without_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_readout(directory, [saved_row(PREPARE_APP)])
            save_actor(directory, artifact=False)
            result = self.analyze(directory)
        self.assertEqual(result['errors'], [])
        link = result['actor_extreme_links'][0]
        self.assertEqual(link['actual_whitened_advantage'], -9.)
        self.assertEqual(link['matching_readout_extremal_slot_records'], [])
        self.assertFalse(link['retained_token_ID_verified'])

    def test_mismatched_selected_ids_report_alignment_error(self):
        saved = saved_row(PREPARE_APP)
        saved['selected'][0] = 999
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_readout(directory, [saved])
            result = self.analyze(directory)
        self.assertEqual(result['counts']['readout_work_rows'], 0)
        self.assertEqual(result['errors'][0]['stage'], 'readout_row')


if __name__ == '__main__':
    unittest.main()
