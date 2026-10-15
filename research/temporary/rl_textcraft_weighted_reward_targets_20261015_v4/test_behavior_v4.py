"""Owner behavior checks; full native GPU/capacity checks are separate."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
BASE=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/baseline'


def original(name):
    spec=importlib.util.spec_from_file_location('v3_original_'+name,BASE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


@pytest.mark.gpu
def test_default_finite_head_is_bitwise_original():
    import qwen35_answer_finite as current
    old=original('qwen35_answer_finite')
    generator=torch.Generator().manual_seed(19)
    head=torch.nn.Linear(16,32,bias=False,device='cuda',dtype=torch.bfloat16)
    logits=torch.randn(8,32,generator=generator).to('cuda')
    case={'target_ids':torch.tensor([3,4,5,6]),'prompt_length':1}
    a=old.PackedAnswerTargets([case],[[0,1,2,3]],5,'cuda')
    b=current.PackedAnswerTargets([case],[[0,1,2,3]],5,'cuda')
    x,xd=old.FiniteAnswerOps(False)(logits,head,a)
    y,yd=current.FiniteAnswerOps(False)(logits,head,b)
    assert torch.equal(x,y)
    assert all(torch.equal(xd[key],yd[key]) for key in xd)


def test_signed_span_budgets_and_zero_difference_behavior():
    from qwen35_answer_finite import PackedAnswerTargets
    # Repeated token IDs must remain distinct target positions/credit groups.
    case={'target_ids':torch.tensor([3,3,3,3]),'prompt_length':1}
    selection=PackedAnswerTargets([case],[[0,1,2,3]],5,'cpu',
        target_span_credits=[[([0,1],-.25),([2,3],1.25)]])
    zero=torch.zeros(4)
    selection.prepare_credit_seed(zero,torch.tensor([1.,1.,2.,2.]))
    assert selection.credit_seed_weights.tolist()==[-.125,-.125,.3125,.3125]
    assert selection.weight_target_values(torch.tensor([1.,1.,2.,2.])).sum()==1
    # Empty endpoint contrast cannot carry a nonzero fixed credit; no epsilon.
    with pytest.raises(FloatingPointError,match='zero/nonfinite'):
        selection.prepare_credit_seed(zero,zero)
    optional=PackedAnswerTargets([case],[[0,1,2,3]],5,'cpu',
        target_span_credits=[[([0,1],0.),([2,3],1.)]])
    optional.prepare_credit_seed(zero,torch.tensor([0.,0.,2.,2.]))
    assert optional.credit_seed_weights.tolist()==[0.,0.,.25,.25]
    with pytest.raises(ValueError,match='partition'):
        PackedAnswerTargets([case],[[0,1,2,3]],5,'cpu',
            target_span_credits=[[([0,1],.5),([1,2,3],.5)]])


def test_linear_credit_sign_masks_self_and_QVA():
    from counterfactual import reward_event_token_credit
    allocation=torch.tensor([[[-2.,.5,0.,100000.]]])
    coefficient=torch.tensor([[-.5]])
    policy=torch.tensor([[True,True,False,True]])
    target=torch.tensor([[[False,True,False,False]]])
    result=reward_event_token_credit(allocation,coefficient,torch.ones_like(allocation,dtype=torch.bool),
                                     policy,self_target_mask=target,linear_attribution=True)
    assert result.advantages.tolist()==[[1.,-.5,0.,-50000.]]
    assert torch.equal(result.q_estimates-result.v_estimates,result.advantages)
    assert result.v_estimates[0,1]==0
    assert all(value[0,2]==0 for value in (result.advantages,result.q_estimates,result.v_estimates))


@pytest.mark.gpu
def test_weighted_seed_keeps_original_bounded_chunks(monkeypatch):
    import qwen35_answer_finite as owner
    monkeypatch.setattr(owner,'_TEXT_TARGET_ROW_LIMIT',2)
    head=torch.nn.Linear(16,32,bias=False,device='cuda',dtype=torch.bfloat16)
    logits=torch.randn(12,32,device='cuda')
    case={'target_ids':torch.tensor([3]*6),'prompt_length':1}
    selection=owner.PackedAnswerTargets([case],[list(range(6))],7,'cuda',
        target_span_credits=[[([0,1],0.),([2,3,4,5],1.)]])
    logp=owner.selected_target_log_probs(logits,selection)
    selection.prepare_credit_seed(logp[0::2],logp[1::2])
    ops=owner.FiniteAnswerOps(False);original_seed=ops.seed;seen=[]
    def observe(*args,**kwargs):
        seen.append(kwargs['multipliers'].clone());return original_seed(*args,**kwargs)
    ops.seed=observe
    _,result=ops(logits,head,selection)
    assert len(seen)==3 and torch.equal(torch.cat(seen),selection.credit_seed_weights)
    assert not bool(result['packed_hidden'][:2].any())


def test_default_credit_is_bitwise_original():
    from counterfactual import reward_event_token_credit
    old=original('counterfactual').reward_event_token_credit
    args=(torch.tensor([[[-.5,.25,0.,1.]]]),torch.tensor([[.75]]),
          torch.ones(1,1,4,dtype=torch.bool),torch.tensor([[True,True,False,True]]))
    a=old(*args);b=reward_event_token_credit(*args)
    for field in ('advantages','q_estimates','v_estimates'):
        assert torch.equal(getattr(a,field),getattr(b,field))


def test_original_native_rollout_zero_settlement_target_only(tmp_path,monkeypatch):
    # Reuse the existing actual-owner/service/RPC fixture. Only observe its
    # original full-trajectory artifacts; no second rollout or parser.
    import executed_target_spans
    import test_textcraft_trajectory_entry as owner_test
    from transformers import AutoTokenizer
    from types import SimpleNamespace
    # The current owner scope additionally calls these original worker RPCs.
    # Its existing CPU engine fixture has no GPU state; retain the calls.
    context_calls=[]
    def transport_namespace(*args,**kwargs):
        value=SimpleNamespace(*args,**kwargs)
        if 'generate_sequences' in kwargs:
            value.begin_rollout_context=lambda:context_calls.append('begin')
            value.end_rollout_context=lambda:context_calls.append('end')
        return value
    monkeypatch.setattr(owner_test,'SimpleNamespace',transport_namespace)
    # Stop this fixture at the original owner return. Its older downstream
    # sentinel-credit assertions predate the official group-multiplier input;
    # credit transport is covered separately, without fabricating that input.
    import textcraft_owner_rollout
    class NativeOutputCollected(Exception):pass
    native_outputs=[]
    original_collect=textcraft_owner_rollout.TextCraftOwner.collect_native_trajectories
    def collect_native(self,*args,**kwargs):
        native_outputs.append(original_collect(self,*args,**kwargs))
        raise NativeOutputCollected()
    monkeypatch.setattr(textcraft_owner_rollout.TextCraftOwner,'collect_native_trajectories',collect_native)
    monkeypatch.setenv('DT_CREDIT_MODE','weighted_reward_target_finite_v4')
    observed=[]
    original_artifact=executed_target_spans.full_trajectory_artifact
    def retain(*args,**kwargs):
        result=original_artifact(*args,**kwargs);observed.append(result);return result
    monkeypatch.setattr(executed_target_spans,'full_trajectory_artifact',retain)
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    with pytest.raises(NativeOutputCollected):
        owner_test.test_real_service_native_trajectory_and_exact_generated_prefix(
            tokenizer,tmp_path,monkeypatch,True,None)
    assert observed
    assert context_calls==['begin','end']
    assert len(native_outputs)==1 and len(observed)==len(native_outputs[0])==16
    assert native_outputs[0].batch['task_rounds'].tolist()==[2.]*16
    assert not bool(native_outputs[0].batch['rm_scores'].any())
    for artifact in observed:
        spans=artifact['reward_target_spans']
        assert len(spans)==1 and spans[0]['reward']==0
        assert spans[0]['action_turn_index']==1
        assert spans[0]['credit_fraction']==1
        assert spans[0]['target_offsets']
        # Earlier Action remains a policy action but is no longer a target.
        assert len(artifact['payload_metadata'])==1
        assert all(artifact['policy_mask'][i] for i in spans[0]['target_offsets'])
        first_target=min(spans[0]['target_offsets'])
        assert any(artifact['policy_mask'][:first_target])
    torch.save(observed,HERE.parent/'native-zero-settlement-artifacts.pt')
