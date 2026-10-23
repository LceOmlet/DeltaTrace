"""Contracts for attribution-before-grouping through real native DataProto/trainer."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo import ray_trainer as trainer
from native_contract_fixtures import make_data, PrimitiveContractWorker
from saved_native_outputs import SavedNativeOutputs, PAIR, digest
import dt_training_batch as credit

HERE = Path(__file__).resolve().parent
MODE = 'reward_attribution_then_group_v12'


def data_fixture():
    data = make_data()
    data.batch.pop('dt_grpo_multipliers')
    reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    reward[:, -2] = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.])
    data.batch['token_level_rewards'] = reward
    data.batch['loss_mask'] = data.batch['response_mask'].clone()
    data.non_tensor_batch['uid'] = np.array(['A']*4+['B']*4)
    return data


def prepare(data):
    return trainer.prepare_dt_credit_multipliers(data, norm_adv_by_std_in_grpo=True, multi_turn=True)


def raw_credit(data, worker=None):
    values = credit.compute_direct_target_credit(data, worker or PrimitiveContractWorker(),
        eos_token_id=248046, pad_token_id=0)
    return data.union(values)


def normalize(data):
    return trainer.compute_advantage(data, trainer.AdvantageEstimator.DELTATRACE, multi_turn=True)


def expected_table_normalization(data):
    raw = data.batch['dt_token_advantages'].double()
    mask = data.batch['response_mask'].bool()
    expected = torch.zeros_like(raw)
    for group in set(data.non_tensor_batch['uid']):
        rows = np.flatnonzero(data.non_tensor_batch['uid'] == group).tolist()
        values = torch.cat([raw[i][mask[i]] for i in rows])
        mean, std = values.mean(), values.std(correction=0)
        for i in rows:
            expected[i, mask[i]] = (raw[i, mask[i]]-mean)/(std+1e-6)
    return expected.float()


def test_v12_does_not_compute_GRPO_before_attribution(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE', MODE)
    data = data_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('v12 must not compute a pre-attribution GRPO g')
    monkeypatch.setattr(trainer, 'compute_advantage', forbidden)
    prepare(data)
    assert torch.equal(data.batch['dt_reward_multipliers'], data.batch['token_level_rewards'].sum(-1))
    assert 'dt_grpo_multipliers' not in data.batch


def test_v11_preparation_and_final_advantages_are_unchanged(monkeypatch):
    spec = importlib.util.spec_from_file_location('baseline_trainer_v11', HERE/'owner-sources/ray_trainer.py')
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    monkeypatch.setenv('DT_CREDIT_MODE', 'reward_target_exponential_v11')
    data = data_fixture()
    expected = base.compute_advantage(data_fixture(), base.AdvantageEstimator.GRPO, multi_turn=True)
    first = data.batch['response_mask'].long().argmax(-1, keepdim=True)
    g = expected.batch['advantages'].gather(1, first).squeeze(1)
    prepare(data)
    assert torch.equal(data.batch['dt_grpo_multipliers'], g)
    before = base.compute_advantage(raw_credit(data), base.AdvantageEstimator.DELTATRACE)
    after = normalize(raw_credit(prepare(data_fixture())))
    assert torch.equal(before.batch['advantages'], after.batch['advantages'])


def test_raw_reward_table_and_shared_baseline_make_zero_reward_negative(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE', MODE)
    data = raw_credit(prepare(data_fixture()))
    zero = data.batch['dt_reward_multipliers'] == 0
    assert not data.batch['dt_token_advantages'][zero].any()
    # Original primitive is used for source expm1 and the constant self target.
    success = ~zero
    assert torch.equal(data.batch['dt_token_advantages'][success, 5], torch.ones(int(success.sum())))
    expected = expected_table_normalization(data)
    out = normalize(data)
    assert torch.equal(out.batch['advantages'], expected)
    assert (out.batch['advantages'][zero][out.batch['response_mask'][zero]] < 0).all()
    assert not out.batch['advantages'][~out.batch['response_mask'].bool()].any()


def table_data(table, mask, uid):
    table = torch.tensor(table, dtype=torch.float32)
    return DataProto.from_dict(tensors=dict(responses=torch.zeros_like(table,dtype=torch.long),
        response_mask=torch.tensor(mask,dtype=torch.bool), dt_token_advantages=table,
        dt_q_estimates=table.clone(),dt_v_estimates=torch.zeros_like(table)),
        non_tensors=dict(uid=np.array(uid)))


def test_one_pool_with_variable_lengths_and_isolated_groups(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    data = table_data([[1,999,999,999],[0,0,0,0],[2,2,999,999]],
        [[1,0,0,0],[1,1,1,1],[1,1,0,0]],['A','A','B'])
    expected = expected_table_normalization(data)
    result = normalize(data)
    assert torch.equal(result.batch['advantages'],expected)
    stats = {x['uid']:x for x in result.meta_info['dt_v12_group_statistics']}
    assert stats['A']['reward_mean'] == .2 and stats['A']['policy_tokens'] == 5
    assert stats['B']['reward_mean'] == 2. and stats['B']['reward_std'] == 0.
    assert not result.batch['advantages'][2].any()


def test_epsilon_placement_and_no_second_batch_STD(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    data=table_data([[1e-9,0],[2e-9,0],[0,0]],[[1,0],[1,0],[1,0]],['A']*3)
    expected=expected_table_normalization(data)
    out=normalize(data)
    assert torch.equal(out.batch['advantages'],expected)
    assert out.batch['advantages'].abs().max() < .01


def test_constant_zero_and_single_policy_token_groups_stay_finite(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    data=table_data([[0,0],[1,1],[2,0],[0,0]],[[1,1],[1,1],[1,0],[0,0]],['zero','one','single','empty'])
    out=normalize(data)
    assert torch.isfinite(out.batch['advantages']).all()
    assert not out.batch['advantages'].any()


class RawRewardSavedOutputs(SavedNativeOutputs):
    """Only rescale actual previously returned DT tensors by the new raw coefficient."""
    def compute_dt_token_advantages(self, requests):
        result=super().compute_dt_token_advantages(requests)
        for i,uid in enumerate(requests.non_tensor_batch['traj_uid']):
            before,_=self.by_uid[str(uid)]
            old=before.batch['dt_direct_reward'][0].double()
            assert old != 0
            ratio=requests.batch['dt_direct_reward'][i].double()/old
            for key in credit.CREDIT_KEYS:
                result.batch[key][i]=(result.batch[key][i].double()*ratio).float()
        return result


def test_saved_real_DT_groups_and_native_full_trainer(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    replay=RawRewardSavedOutputs()
    paths=[PAIR/f/'002-update_actor-input.pkl' for f in ['rank0-pid1021020','rank1-pid1023296']]
    original=DataProto.concat([DataProto.load_from_disk(str(p)) for p in paths])
    rewards=original.batch['token_level_rewards'].sum(-1)
    # Old all-success groups had g=0 and no saved DT call. Do not invent those d.
    unavailable={original.non_tensor_batch['uid'][i] for i in range(len(original))
        if rewards[i] != 0 and str(original.non_tensor_batch['traj_uid'][i]) not in replay.by_uid}
    rows=[i for i,g in enumerate(original.non_tensor_batch['uid']) if g not in unavailable]
    data=original.select_idxs(rows)
    assert len(data)>0
    data.batch.pop('dt_grpo_multipliers')
    # Saved update_actor input already carries the previous method's credit.
    for key in credit.CREDIT_KEYS+('advantages','returns'):
        if key in data.batch: data.batch.pop(key)
    data=raw_credit(prepare(data),replay)
    raw=data.batch['dt_token_advantages'].clone()
    expected=expected_table_normalization(data)
    out=normalize(data)
    assert torch.equal(out.batch['advantages'],expected)
    mask=out.batch['response_mask'].bool()
    for name in credit.CREDIT_KEYS+('advantages','returns'):
        assert torch.isfinite(out.batch[name]).all()
        assert not out.batch[name][~mask].any()
    zero=out.batch['dt_reward_multipliers']==0
    stats=out.meta_info['dt_v12_group_statistics']
    means={x['uid']:x['reward_mean'] for x in stats}
    for i in range(len(out)):
        if zero[i] and means[str(out.non_tensor_batch['uid'][i])]>0:
            assert (out.batch['advantages'][i][mask[i]]<0).all()
    v=out.batch['advantages'][mask].double()
    record=dict(status='passed_saved_real_tensor_native_trainer_integration',
        original_rows=len(original),complete_available_group_rows=len(data),
        groups=len(stats),unavailable_old_all_success_group_ids=[str(x) for x in unavailable],
        unavailable_reason='Old g=0 traces were never computed; no new model or fabricated d',
        raw_table_min=float(raw[mask].min()),raw_table_max=float(raw[mask].max()),
        final_A_min=float(v.min()),final_A_max=float(v.max()),final_A_RMS=float(v.square().mean().sqrt()),
        max_Q_minus_V_minus_A=float((out.batch['dt_q_estimates']-out.batch['dt_v_estimates']-out.batch['advantages'])[mask].abs().max()),
        group_statistics=stats,epsilon=1e-6,std_correction=0,second_normalization=False,
        model_DT_optimizer_environment_calls=0,sources=replay.sources+[digest(p) for p in paths])
    (HERE.parent/'saved-native-v12-verification.json').write_text(json.dumps(record,indent=2)+'\n')
