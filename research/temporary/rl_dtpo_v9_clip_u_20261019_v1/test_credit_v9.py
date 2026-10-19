"""v9 changes only per-event exponential u; call native transport and variance."""
import functools
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
sys.path.append(str(ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source'))
sys.path.append(str(ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source'))
from counterfactual import reward_event_token_credit
from dt_training_batch import compute_direct_target_credit,CREDIT_KEYS
from test_direct_target_transport import artifact,native_training_data
import test_multi_target_v5 as transport

MODE='reward_target_exponential_clipped_v9'

@pytest.fixture(autouse=True)
def mode(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE',MODE)
    monkeypatch.delenv('DT_NATIVE_HOST_CACHE_BATCH',raising=False)
    monkeypatch.setattr(transport,'reward_event_token_credit',functools.partial(
        reward_event_token_credit,clip_exponential_credit=True))

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('g',[-3.,.25,2.5])
def test_clip_before_signed_coefficient_including_extreme_finite_d(dtype,g):
    d=torch.tensor([[[-10000.,-10.,-1.,-.5,0.,.5,10000.]]],dtype=dtype)
    b=torch.tensor([[g]],dtype=dtype);policy=torch.ones((1,7),dtype=torch.bool)
    output=reward_event_token_credit(d,b,policy[:,None],policy,clip_exponential_credit=True)
    u=(-torch.expm1(-d[:,0])).clamp(-1,1)
    assert torch.equal(output.advantages,b*u)
    assert torch.equal(output.q_estimates,b.expand(1,7))
    assert output.v_estimates[0,0]==2*g
    torch.testing.assert_close(output.q_estimates-output.v_estimates,output.advantages)
    assert bool(torch.isfinite(output.advantages).all())
    assert output.advantages[0,0]==-g  # Not clip(g*u), including |g| > 1.
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_interior_is_bitwise_original_v5_and_default_remains_original(dtype):
    parent=ROOT/'candidates/textcraft-dt-symmetric-probability-ratio-postgroup-variance-20261018-v8/entry/counterfactual.py'
    spec=importlib.util.spec_from_file_location('original_v5_credit_parent',parent)
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    d=torch.tensor([[[-math.log(2)+.001,-.25,0.,1.,30.]]],dtype=dtype)
    b=torch.tensor([[-1.5]],dtype=dtype);p=torch.ones((1,5),dtype=torch.bool)
    expected=original.reward_event_token_credit(d,b,p[:,None],p)
    for kwargs in ({},{'clip_exponential_credit':True}):
        actual=reward_event_token_credit(d,b,p[:,None],p,**kwargs)
        for key in ('advantages','q_estimates','v_estimates'):
            assert torch.equal(getattr(actual,key),getattr(expected,key))
    with pytest.raises(FloatingPointError):
        reward_event_token_credit(torch.full_like(d,-10000),b,p[:,None],p)

def test_self_observation_future_zero_and_discount_semantics():
    d=torch.tensor([[[float('nan'),float('nan'),-1000.,float('nan')]],[[float('nan')]*4]])
    b=torch.tensor([[.75],[0.]])
    p=torch.tensor([[True,False,True,True],[True]*4]);f=torch.tensor([[[True,True,True,False]],[[True]*4]])
    own=torch.tensor([[[True,False,False,False]],[[False]*4]])
    discounts=torch.tensor([[[.5,1.,.25,1.]],[[1.]*4]])
    out=reward_event_token_credit(d,b,f,p,self_target_mask=own,discounts=discounts,clip_exponential_credit=True)
    assert (out.advantages[0,0],out.q_estimates[0,0],out.v_estimates[0,0])==(.375,.375,0.)
    assert (out.advantages[0,2],out.q_estimates[0,2],out.v_estimates[0,2])==(-.1875,.1875,.375)
    for value in (out.advantages,out.q_estimates,out.v_estimates):
        assert value[0,1]==value[0,3]==0 and torch.equal(value[1],torch.zeros(4))

@pytest.mark.parametrize('credits',[(-.1,1.),(-1.,1.)])
def test_clip_individual_events_before_sum_and_keep_signed_cancellation(credits):
    d=torch.tensor([[[-3.,.2],[2.,-4.]]]);b=torch.tensor([credits]);p=torch.ones((1,2),dtype=torch.bool)
    out=reward_event_token_credit(d,b,p[:,None].expand_as(d),p,clip_exponential_credit=True)
    expected=(b[:,:,None]*(-torch.expm1(-d)).clamp(-1,1)).sum(1)
    assert torch.equal(out.advantages,expected)
    assert not torch.equal(out.advantages,(-torch.expm1(-d.sum(1))).clamp(-1,1)*sum(credits))
    torch.testing.assert_close(out.q_estimates-out.v_estimates,out.advantages)
    transport.test_multitarget_DP_preserves_signed_components_and_retained_slots(credits)

def test_original_native_multi_event_and_retained_slot_contracts():
    transport.test_earlier_target_is_source_for_later_event_and_future_masks_are_retained()
    transport.test_multiple_payload_pieces_of_one_event_are_not_credited_twice()
    transport.test_mixed_event_counts_and_zero_component_keep_native_equal_rank_calls()

@pytest.mark.parametrize('multipliers',[[.25,-1.5,0.],[.5,-3.,0.]])
def test_native_post_g_variance_QVA_exactly_once_without_centering(monkeypatch,multipliers):
    from verl.utils import torch_functional as vf
    from verl.trainer.ppo.ray_trainer import compute_advantage,AdvantageEstimator
    original=vf.masked_var;calls=[]
    def measured(values,mask,*args,**kwargs):
        calls.append((values.clone(),mask.clone()));return original(values,mask,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('No mean subtraction or pre-g whitening')
    monkeypatch.setattr(vf,'masked_var',measured);monkeypatch.setattr(vf,'masked_whiten',forbidden)
    data=native_training_data([artifact(i) for i in range(3)],multipliers=multipliers)
    worker=transport.EventTransportWorker()
    output=compute_direct_target_credit(data,worker,eos_token_id=99,pad_token_id=0)
    assert len(worker.returned)==2 and not calls
    mask=torch.tensor([[False if pos<0 else item['policy_mask'][pos]
        for pos in item['retained_response_positions']] for item in data.non_tensor_batch['dt_direct_target_artifact']])
    data.batch['response_mask']=mask
    raw={key:torch.where(mask,output.batch[key],0.).clone() for key in CREDIT_KEYS}
    scale=torch.rsqrt(original(raw['dt_token_advantages'],mask)+1e-8)
    result=compute_advantage(data.union(output),AdvantageEstimator.DELTATRACE)
    assert len(calls)==1 and torch.equal(calls[0][0],raw['dt_token_advantages']) and torch.equal(calls[0][1],mask)
    for key in CREDIT_KEYS:assert torch.equal(result.batch[key],torch.where(mask,raw[key]*scale,0.))
    assert torch.equal(result.batch['advantages'],result.batch['dt_token_advantages'])
    assert torch.equal(result.batch['returns'],result.batch['dt_q_estimates'])
    torch.testing.assert_close(result.batch['dt_q_estimates']-result.batch['dt_v_estimates'],result.batch['advantages'])

def test_actual_readout_mode_clips_sources_and_keeps_original_EOS_target(monkeypatch):
    import reward_readout
    item=artifact(0);response=torch.tensor(item['response_ids'])
    prompt=torch.tensor(item['prompt_ids']);ids=torch.cat((prompt,response))
    row=dict(traj_uid='readout-v9',input_ids=ids,attention_mask=torch.ones_like(ids),responses=response,
        policy_mask=torch.tensor(item['policy_mask']),target_mask=torch.tensor(item['target_mask']),dt_direct_reward=torch.tensor(-.25))
    captured=[]
    def native_saved_effects(runner,reference,factual,cases,targets,**kwargs):
        captured.append((reference.clone(),factual.clone()))
        return torch.full_like(factual,-2.,dtype=torch.float32),None,dict(root_effect=-2.,
            policy_credit_signed_sum=-2.,conservation_residual=0.,factual_target_logp=-3.,reference_target_logp=-1.)
    monkeypatch.setattr(reward_readout,'trace_token_attribution',native_saved_effects)
    runner=SimpleNamespace(model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros(1))))
    readout=reward_readout.DirectActionTargetReadout(runner,SimpleNamespace(eos_token_id=99),task='TextCraft',packed_answer_targets=None)
    result,=readout.trajectories([row])
    assert len(captured)==1 and readout.clip_exponential_credit and readout.reward_target_exponential
    assert not readout.weighted_target_credit and not readout.symmetric_probability_ratio
    assert readout.last_report['credit_transform']=='per_event_clipped_exponential_then_credit_sum'
    assert readout.last_report['event_u_clip']==[-1.,1.]
    policy=row['policy_mask'];target=row['target_mask'];prior=policy&~target
    positions=torch.arange(len(response));prior&=positions<positions[target].max()
    assert torch.equal(result['dt_token_advantages'][prior],torch.full((int(prior.sum()),),.25))
    assert torch.equal(result['dt_token_advantages'][target],torch.full((int(target.sum()),),-.25))
    assert not bool(result['dt_token_advantages'][~policy].any())

def test_v9_native_zero_settlement_target_identity(tmp_path,monkeypatch):
    from test_behavior_v5 import _native_artifacts
    artifacts=_native_artifacts(tmp_path,monkeypatch,MODE)
    for item in artifacts:
        assert len(item['reward_target_spans'])==len(item['payload_metadata'])==1
        assert item['reward_target_spans'][0]['reward']==0
        assert item['reward_target_spans'][0]['action_turn_index']==1

def test_saved_v5_rollout_reproduces_clip_and_preserves_source_sign():
    path=ROOT/'runs/textcraft-dt-reward-target-exponential-20261016-v5/credit-records/rank0-pid888009/joint-1792102314203492009.pt'
    row=torch.load(path,map_location='cpu',weights_only=False)['rows'][2]
    prior=row['prior_source_mask'].bool();d=row['source_log_ratios'][prior][None,None]
    b=torch.tensor([[row['reward']]],dtype=torch.float32);p=torch.ones((1,d.shape[-1]),dtype=torch.bool)
    raw=reward_event_token_credit(d,b,p[:,None],p)
    clipped=reward_event_token_credit(d,b,p[:,None],p,clip_exponential_credit=True)
    expected=b*(-torch.expm1(-d[:,0])).clamp(-1,1)
    assert torch.equal(clipped.advantages,expected)
    assert torch.equal(raw.advantages.sign(),clipped.advantages.sign())
    assert clipped.advantages.min()==-b[0,0] and raw.advantages.min() < -b[0,0]

def test_exact_B4_32k_credit_extent_and_default_mode_isolation():
    d=torch.linspace(-1000,1000,32768).reshape(1,1,-1).repeat(4,2,1)
    b=torch.tensor([[-.1,1.]]).repeat(4,1);p=torch.ones((4,32768),dtype=torch.bool)
    out=reward_event_token_credit(d,b,torch.ones_like(d,dtype=torch.bool),p,clip_exponential_credit=True)
    assert torch.equal(out.advantages,(b[:,:,None]*(-torch.expm1(-d)).clamp(-1,1)).sum(1))
    assert bool(torch.isfinite(out.advantages).all()) and not torch.cuda.is_initialized()

@pytest.mark.parametrize('kwargs',[dict(linear_attribution=True),dict(symmetric_probability_ratio=True)])
def test_no_accidental_combination_of_different_credit_transforms(kwargs):
    with pytest.raises(ValueError,match='mutually exclusive'):
        reward_event_token_credit(torch.zeros(1,1,2),torch.ones(1,1),torch.ones(1,1,2,dtype=torch.bool),
            torch.ones(1,2,dtype=torch.bool),clip_exponential_credit=True,**kwargs)
