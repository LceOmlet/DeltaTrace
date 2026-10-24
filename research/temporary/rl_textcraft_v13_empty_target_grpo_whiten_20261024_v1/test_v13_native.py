"""v13 contracts through pinned DataProto, GRPO, DT primitive and trainer."""
import importlib.util
import json
import os
from pathlib import Path

import torch
from verl import DataProto
from verl.trainer.ppo import ray_trainer as trainer
from verl.utils import torch_functional as verl_F
from native_contract_fixtures import make_data, PrimitiveContractWorker
from saved_native_outputs import SavedNativeOutputs, PAIR, digest
import dt_training_batch as credit

HERE = Path(__file__).resolve().parent
MODE = 'reward_target_exponential_whiten_v13'


def normalize(data):
    return trainer.compute_advantage(data, trainer.AdvantageEstimator.DELTATRACE, multi_turn=True)


def raw_credit(data, worker=None):
    return data.union(credit.compute_direct_target_credit(data, worker or PrimitiveContractWorker(),
        eos_token_id=248046, pad_token_id=0))


def check_native_whitening(data):
    mask = data.batch['response_mask'].bool()
    raw = {k:data.batch[k].clone() for k in credit.CREDIT_KEYS}
    expected = torch.where(mask, verl_F.masked_whiten(raw['dt_token_advantages'],mask),0.)
    mean = verl_F.masked_mean(raw['dt_token_advantages'],mask)
    scale = torch.rsqrt(verl_F.masked_var(raw['dt_token_advantages'],mask)+1e-8)
    out = normalize(data)
    assert torch.equal(out.batch['advantages'],expected)
    assert torch.equal(out.batch['dt_token_advantages'],expected)
    assert torch.equal(out.batch['dt_q_estimates'],torch.where(mask,raw['dt_q_estimates']*scale,0.))
    assert torch.equal(out.batch['dt_v_estimates'],torch.where(mask,(raw['dt_v_estimates']+mean)*scale,0.))
    assert torch.equal(out.batch['returns'],out.batch['dt_q_estimates'])
    torch.testing.assert_close(out.batch['dt_q_estimates']-out.batch['dt_v_estimates'],expected)
    for k in credit.CREDIT_KEYS+('advantages','returns'):
        assert torch.isfinite(out.batch[k]).all()
        assert not out.batch[k][~mask].any()
    return out,raw,mean,scale


def test_default_v5_and_v11_trainer_paths_unchanged(monkeypatch):
    spec=importlib.util.spec_from_file_location('v13_parent_trainer',HERE/'owner-sources/ray_trainer.py')
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    for mode in ['reward_target_exponential_v5','reward_target_exponential_v11']:
        monkeypatch.setenv('DT_CREDIT_MODE',mode)
        a=parent.compute_advantage(raw_credit(make_data()),parent.AdvantageEstimator.DELTATRACE)
        b=normalize(raw_credit(make_data()))
        assert all(torch.equal(a.batch[k],b.batch[k]) for k in credit.CREDIT_KEYS+('advantages','returns'))


def test_v13_raw_credit_equals_v11_before_whitening(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    a=raw_credit(make_data(duplicate=True))
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    b=raw_credit(make_data(duplicate=True))
    assert all(torch.equal(a.batch[k],b.batch[k]) for k in credit.CREDIT_KEYS)
    check_native_whitening(b)


def test_centering_includes_raw_zero_policy_and_excludes_observation_padding(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    table=torch.tensor([[2.,0.,999.,0.],[4.,0.,999.,0.]])
    mask=torch.tensor([[1,1,0,0],[1,1,0,0]],dtype=torch.bool)
    data=DataProto.from_dict(tensors=dict(responses=torch.zeros_like(table,dtype=torch.long),response_mask=mask,
        dt_token_advantages=table.clone(),dt_q_estimates=table.clone(),dt_v_estimates=torch.zeros_like(table)))
    out,_,mean,_=check_native_whitening(data)
    assert mean==1.5
    assert (out.batch['advantages'][:,1]<0).all()
    assert abs(float(verl_F.masked_mean(out.batch['advantages'],mask)))<1e-6
    torch.testing.assert_close(verl_F.masked_var(out.batch['advantages'],mask),torch.tensor(1.))


def test_whitening_is_one_whole_batch_not_per_group_or_channel(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    data=make_data()
    data=raw_credit(data)
    # Native UID is irrelevant to final batch whitening; it already determined g.
    first=data.select_idxs([0,1,2,3]);second=data.select_idxs([4,5,6,7])
    together,_,_,_=check_native_whitening(data)
    a,_,_,_=check_native_whitening(first);b,_,_,_=check_native_whitening(second)
    assert not torch.equal(together.batch['advantages'],torch.cat([a.batch['advantages'],b.batch['advantages']]))


def test_all_zero_credit_is_finite_and_stays_zero(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    data=make_data();data.batch['dt_grpo_multipliers'].zero_()
    out,_,_,_=check_native_whitening(raw_credit(data))
    assert not out.batch['advantages'].any()


def test_saved_real_DT_native_scatter_and_final_batch_whitening(monkeypatch):
    replay=SavedNativeOutputs()
    paths=[PAIR/f/'002-update_actor-input.pkl' for f in ['rank0-pid1021020','rank1-pid1023296']]
    data=DataProto.concat([DataProto.load_from_disk(str(p)) for p in paths])
    for key in credit.CREDIT_KEYS+('advantages','returns'):
        if key in data.batch:data.batch.pop(key)
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    v11=raw_credit(data,replay)
    prior={k:v11.batch[k].clone() for k in credit.CREDIT_KEYS}
    for key in credit.CREDIT_KEYS:data.batch.pop(key)
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    v13=raw_credit(data,replay)
    assert all(torch.equal(prior[k],v13.batch[k]) for k in credit.CREDIT_KEYS)
    out,raw,mean,scale=check_native_whitening(v13)
    mask=out.batch['response_mask'].bool();v=out.batch['advantages'][mask].double()
    target=torch.tensor([any(z['target_mask']) for z in out.non_tensor_batch['dt_direct_target_artifact']])
    record=dict(status='passed_saved_real_tensor_native_trainer_integration',
        scope='Old DP2 natural iteration6; not fresh DP8 quality or stability',rows=len(data),
        targeted=int(target.sum()),target_free=int((~target).sum()),
        nonzero_g_target_free=int(((~target)&(out.batch['dt_grpo_multipliers']!=0)).sum()),
        raw_v11_QVA_bitwise_equal=True,raw_batch_mean=float(mean),whitening_scale=float(scale),
        final_A_min=float(v.min()),final_A_max=float(v.max()),final_A_mean=float(v.mean()),
        final_A_RMS=float(v.square().mean().sqrt()),
        max_Q_minus_V_minus_A=float((out.batch['dt_q_estimates']-out.batch['dt_v_estimates']-out.batch['advantages'])[mask].abs().max()),
        shift_mean=True,second_normalization=False,new_model_DT_optimizer_environment_calls=0,
        sources=replay.sources+[digest(p) for p in paths])
    (HERE.parent/'saved-native-v13-verification.json').write_text(json.dumps(record,indent=2)+'\n')
