"""Test the declared v7 seam with native torch and the existing VERL DP fixtures.

No prediction accuracy or training quality is inferred from these CPU tests.
"""
from copy import deepcopy
import functools
import os
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
sys.path.append(str(ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source'))
sys.path.append(str(ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source'))
from counterfactual import reward_event_token_credit
from dt_training_batch import compute_direct_target_credit,CREDIT_KEYS
from test_direct_target_transport import artifact,native_training_data
import test_multi_target_v5 as transport

MODE='reward_target_symmetric_probability_ratio_v7'

@pytest.fixture(autouse=True)
def mode(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    monkeypatch.delenv('DT_NATIVE_HOST_CACHE_BATCH',raising=False)
    monkeypatch.setattr(transport,'reward_event_token_credit',functools.partial(
        reward_event_token_credit,symmetric_probability_ratio=True))

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_direct_native_transform_extremes_and_signed_coefficients(dtype):
    d=torch.tensor([[[-10000.,-11.873,-1.,0.,1.,11.873,10000.]]],dtype=dtype)
    r=torch.tensor([[-1.5]],dtype=dtype);p=torch.ones((1,7),dtype=torch.bool)
    out=reward_event_token_credit(d,r,p[:,None],p,symmetric_probability_ratio=True)
    assert torch.equal(out.advantages,r*torch.tanh(d[:,0]*.5))
    assert torch.equal(out.q_estimates,r*torch.sigmoid(d[:,0]))
    assert torch.equal(out.v_estimates,r*torch.sigmoid(-d[:,0]))
    assert torch.isfinite(out.advantages).all() and (out.advantages.abs()<=r.abs()).all()
    assert torch.equal(torch.tanh(d*.5),-torch.tanh(-d*.5))
    assert not torch.cuda.is_initialized()

def test_self_future_observation_and_zero_coefficient_masks():
    d=torch.tensor([[[float('nan'),float('nan'),-11.873,float('nan')]],[[float('nan')]*4]])
    r=torch.tensor([[.75],[0.]]);p=torch.tensor([[True,False,True,True],[True]*4])
    f=torch.tensor([[[True,True,True,False]],[[True]*4]])
    own=torch.tensor([[[True,False,False,False]],[[False]*4]])
    out=reward_event_token_credit(d,r,f,p,self_target_mask=own,symmetric_probability_ratio=True)
    assert (out.advantages[0,0],out.q_estimates[0,0],out.v_estimates[0,0])==(.75,.75,0.)
    for v in (out.advantages,out.q_estimates,out.v_estimates):
        assert v[0,1]==v[0,3]==0 and torch.equal(v[1],torch.zeros(4))

@pytest.mark.parametrize('credits',[(-.1,1.),(-1.,1.)])
def test_multi_target_native_DP_signed_components_no_normalization(credits):
    # Reuse the original behavior test and its real native dispatch/collect.
    transport.test_multitarget_DP_preserves_signed_components_and_retained_slots(credits)

def test_multi_target_earlier_target_source_and_future_masks():
    transport.test_earlier_target_is_source_for_later_event_and_future_masks_are_retained()

@pytest.mark.parametrize('multipliers',[[.25,-1.5,0.],[.5,-3.,0.]])
def test_driver_and_native_trainer_never_normalize_credit(monkeypatch,multipliers):
    from verl.utils import torch_functional as vf
    from verl.trainer.ppo.ray_trainer import compute_advantage,AdvantageEstimator
    def forbidden(*args,**kwargs):raise AssertionError('v7 must not normalize credit')
    monkeypatch.setattr(vf,'masked_var',forbidden)
    monkeypatch.setattr(vf,'masked_whiten',forbidden)
    data=native_training_data([artifact(i) for i in range(3)],multipliers=multipliers)
    worker=transport.EventTransportWorker()
    output=compute_direct_target_credit(data,worker,eos_token_id=99,pad_token_id=0)
    assert len(worker.returned)==2  # g=0 requires no DT when there is no batch u statistic.
    for row,item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        for key in CREDIT_KEYS:
            expected=torch.zeros_like(output.batch[key][row])
            if multipliers[row]:
                values=worker.returned[str(data.non_tensor_batch['traj_uid'][row])][key]
                for slot,pos in enumerate(item['retained_response_positions']):
                    if pos>=0:expected[slot]=values[pos]
            assert torch.equal(output.batch[key][row],expected)
    mask=torch.tensor([[False if pos<0 else item['policy_mask'][pos]
        for pos in item['retained_response_positions']] for item in data.non_tensor_batch['dt_direct_target_artifact']])
    data.batch['response_mask']=mask
    result=compute_advantage(data.union(output),AdvantageEstimator.DELTATRACE)
    assert torch.equal(result.batch['advantages'],output.batch['dt_token_advantages'])
    assert torch.equal(result.batch['returns'],output.batch['dt_q_estimates'])
    assert not torch.cuda.is_initialized()

def test_empty_source_analytic_self_path_and_native_no_RPC():
    item=artifact(0);item['target_mask']=list(item['policy_mask'])
    data=native_training_data([item],multipliers=[-.25]);worker=transport.EventTransportWorker()
    out=compute_direct_target_credit(data,worker,eos_token_id=99,pad_token_id=0)
    assert not worker.calls
    for slot,pos in enumerate(item['retained_response_positions']):
        expected=-.25 if pos>=0 and item['policy_mask'][pos] else 0.
        assert out.batch['dt_token_advantages'][0,slot]==expected
        assert out.batch['dt_v_estimates'][0,slot]==0

def test_exact_32k_tensor_extent_uses_bounded_native_ops():
    d=torch.linspace(-1000,1000,32768).reshape(1,1,-1).repeat(4,2,1)
    r=torch.tensor([[-.1,1.]]).repeat(4,1)
    p=torch.ones((4,32768),dtype=torch.bool);f=torch.ones_like(d,dtype=torch.bool)
    out=reward_event_token_credit(d,r,f,p,symmetric_probability_ratio=True)
    expected=(r[:,:,None]*torch.tanh(d*.5)).sum(1)
    assert torch.equal(out.advantages,expected) and torch.isfinite(out.advantages).all()
    assert not torch.cuda.is_initialized()  # Tensor-extent behavior, not another GPU capacity claim.
