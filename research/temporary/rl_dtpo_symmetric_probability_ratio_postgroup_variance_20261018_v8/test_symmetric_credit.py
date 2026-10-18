"""Test the declared v8 seam with native torch and the existing VERL DP fixtures.

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

MODE='reward_target_symmetric_probability_ratio_postgroup_variance_v8'

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
def test_native_trainer_scales_post_group_sum_QVA_once_without_centering(monkeypatch,multipliers):
    from verl.utils import torch_functional as vf
    from verl.trainer.ppo.ray_trainer import compute_advantage,AdvantageEstimator
    original_var=vf.masked_var
    calls=[]
    def measured(values,mask,*args,**kwargs):
        calls.append((values.clone(),mask.clone()))
        return original_var(values,mask,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('no mean-subtract/add-back whitening')
    monkeypatch.setattr(vf,'masked_var',measured)
    monkeypatch.setattr(vf,'masked_whiten',forbidden)
    data=native_training_data([artifact(i) for i in range(3)],multipliers=multipliers)
    worker=transport.EventTransportWorker()
    output=compute_direct_target_credit(data,worker,eos_token_id=99,pad_token_id=0)
    assert len(worker.returned)==2 and not calls  # no pre-g normalization, zero g skips RPC
    mask=torch.tensor([[False if pos<0 else item['policy_mask'][pos]
        for pos in item['retained_response_positions']] for item in data.non_tensor_batch['dt_direct_target_artifact']])
    data.batch['response_mask']=mask
    raw={key:torch.where(mask,output.batch[key],0.).clone() for key in CREDIT_KEYS}
    scale=torch.rsqrt(original_var(raw['dt_token_advantages'],mask)+1e-8)
    result=compute_advantage(data.union(output),AdvantageEstimator.DELTATRACE)
    assert len(calls)==1 and torch.equal(calls[0][0],raw['dt_token_advantages'])
    assert torch.equal(calls[0][1],mask)
    for key in CREDIT_KEYS:
        assert torch.equal(result.batch[key],torch.where(mask,raw[key]*scale,0.))
    assert torch.equal(result.batch['advantages'],result.batch['dt_token_advantages'])
    assert torch.equal(result.batch['returns'],result.batch['dt_q_estimates'])
    torch.testing.assert_close(result.batch['dt_q_estimates']-result.batch['dt_v_estimates'],result.batch['advantages'])
    assert torch.equal(result.batch['advantages'].sign(),raw['dt_token_advantages'].sign())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('credits',[(-.1,1.),(-1.,1.)])
def test_multitarget_scaling_uses_final_event_sum_not_individual_events(credits):
    from verl.utils import torch_functional as vf
    from verl.trainer.ppo.ray_trainer import compute_advantage,AdvantageEstimator
    # The original native transport test above separately verifies event ownership.
    # At this trainer seam, use the actual multi-event native primitive output.
    d=torch.tensor([[[.1,-.3,2.,0.],[1.,-.1,-2.,0.]]])
    mask=torch.tensor([[True,True,True,False]])
    out=reward_event_token_credit(d,torch.tensor([credits]),torch.ones_like(d,dtype=torch.bool),mask,symmetric_probability_ratio=True)
    from verl import DataProto
    data=DataProto.from_dict(tensors=dict(responses=torch.zeros((1,4),dtype=torch.long),response_mask=mask,
        dt_token_advantages=out.advantages,dt_q_estimates=out.q_estimates,dt_v_estimates=out.v_estimates))
    scale=torch.rsqrt(vf.masked_var(out.advantages,mask)+1e-8)
    result=compute_advantage(data,AdvantageEstimator.DELTATRACE)
    assert torch.equal(result.batch['advantages'],out.advantages*scale)
    assert result.batch['advantages'][0,-1]==0


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
