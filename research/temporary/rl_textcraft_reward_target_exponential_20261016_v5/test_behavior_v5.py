"""Contracts of the new seam; all owner algorithms are called directly."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
sys.path.append(str(ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source'))


def test_original_exponential_credit_is_bitwise_preserved():
    from counterfactual import reward_event_token_credit
    spec=importlib.util.spec_from_file_location('v5_accepted_credit_baseline',AUDIT/'baseline/counterfactual.py')
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    args=(torch.tensor([[[-.5,.25,0.,1.]],[[.125,-.75,1.,0.]]]),
          torch.tensor([[.75],[-.5]]),torch.ones(2,1,4,dtype=torch.bool),
          torch.tensor([[True,True,False,True],[True,True,True,False]]))
    left=original.reward_event_token_credit(*args)
    right=reward_event_token_credit(*args)
    for name in ('advantages','q_estimates','v_estimates'):
        assert torch.equal(getattr(left,name),getattr(right,name))


def test_signed_multi_event_credit_exponentiates_before_sum_without_u_normalization():
    from counterfactual import reward_event_token_credit
    d=torch.tensor([[[0.,.25,-.5],[.75,-.25,0.]]],dtype=torch.float64)
    b=torch.tensor([[-.1,1.]],dtype=torch.float64)
    credit=reward_event_token_credit(d,b,torch.ones_like(d,dtype=torch.bool),torch.ones(1,3,dtype=torch.bool))
    expected=(-.1*-torch.expm1(-d[:,0,:]))+(-torch.expm1(-d[:,1,:]))
    assert torch.equal(credit.advantages,expected)
    assert torch.equal(credit.q_estimates,torch.full((1,3),.9,dtype=torch.float64))
    # A zero total observed coefficient must not erase differing target effects.
    cancelled=reward_event_token_credit(d,torch.tensor([[-1.,1.]],dtype=torch.float64),
        torch.ones_like(d,dtype=torch.bool),torch.ones(1,3,dtype=torch.bool))
    assert not bool(cancelled.q_estimates.any()) and bool(cancelled.advantages.any())
    assert not torch.equal(credit.advantages,-torch.expm1(-d.sum(1))*.9)
    assert float(credit.advantages.sum()) != .9


def test_observation_padding_self_target_and_causal_masks():
    from counterfactual import reward_event_token_credit
    d=torch.tensor([[[float('nan'),-.5,.25,1000.],[float('nan'),1.,float('nan'),1000.]]])
    policy=torch.tensor([[False,True,True,False]])
    future=torch.tensor([[[False,True,True,False],[False,True,False,False]]])
    own=torch.tensor([[[False,False,True,False],[False,True,False,False]]])
    b=torch.tensor([[-.25,1.25]])
    result=reward_event_token_credit(d,b,future,policy,self_target_mask=own)
    assert result.advantages[0,2]==-.25 and result.v_estimates[0,2]==0
    for value in (result.advantages,result.q_estimates,result.v_estimates):
        assert not bool(value[~policy].any()) and bool(torch.isfinite(value).all())
    assert result.q_estimates[0,1]==1


def _native_artifacts(tmp_path,monkeypatch,mode):
    from transformers import AutoTokenizer
    from types import SimpleNamespace
    import test_textcraft_trajectory_entry as owner_test
    import textcraft_owner_rollout
    def namespace(*args,**kwargs):
        result=SimpleNamespace(*args,**kwargs)
        if 'generate_sequences' in kwargs:
            result.begin_rollout_context=lambda:None
            result.end_rollout_context=lambda:None
        return result
    monkeypatch.setattr(owner_test,'SimpleNamespace',namespace)
    outputs=[]
    native=textcraft_owner_rollout.TextCraftOwner.collect_native_trajectories
    class Collected(Exception):pass
    def collect(self,*args,**kwargs):
        outputs.append(native(self,*args,**kwargs));raise Collected()
    monkeypatch.setattr(textcraft_owner_rollout.TextCraftOwner,'collect_native_trajectories',collect)
    if mode is None:monkeypatch.delenv('DT_CREDIT_MODE',raising=False)
    else:monkeypatch.setenv('DT_CREDIT_MODE',mode)
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    with pytest.raises(Collected):
        owner_test.test_real_service_native_trajectory_and_exact_generated_prefix(tokenizer,tmp_path,monkeypatch,True,None)
    assert len(outputs)==1 and len(outputs[0])==16
    assert outputs[0].batch['task_rounds'].tolist()==[2.]*16
    assert not bool(outputs[0].batch['rm_scores'].any())
    return list(outputs[0].non_tensor_batch['dt_direct_target_artifact'])


def test_native_zero_settlement_selects_only_its_original_action(tmp_path,monkeypatch):
    artifacts=_native_artifacts(tmp_path,monkeypatch,'reward_target_exponential_v5')
    for item in artifacts:
        event=item['reward_target_spans'][0]
        assert len(item['reward_target_spans'])==len(item['payload_metadata'])==1
        assert event['reward']==0 and event['action_turn_index']==1
        assert any(item['policy_mask'][:min(event['target_offsets'])])
        assert all(item['policy_mask'][i] for i in event['target_offsets'])
    torch.save(artifacts,AUDIT/'native-zero-settlement-artifacts.pt')


def test_original_two_payloads_supply_exact_multitarget_offsets(tmp_path,monkeypatch):
    artifacts=_native_artifacts(tmp_path,monkeypatch,None)
    for item in artifacts:
        assert len(item['payload_metadata'])==2
        events=[]
        for event_index,metadata in enumerate(item['payload_metadata']):
            start=metadata['start']-len(item['prompt_ids'])
            offsets=sorted({start+i for span in metadata['token_mapping']['ranges']
                            for i in range(*span['token_span'])})
            assert offsets and all(item['policy_mask'][i] for i in offsets)
            events.append(dict(event_id=event_index,target_offsets=offsets,
                credit=(-.1,1.)[event_index],
                future_policy_mask=[i<=max(offsets) for i in range(len(item['response_ids']))]))
        assert set(i for event in events for i in event['target_offsets'])=={
            i for i,t in enumerate(item['target_mask']) if t}
        item['dt_reward_target_events']=events
    # Actual native parser offsets and histories; the two user-given credits
    # are an interface fixture, not a claim of two TextCraft settlements.
    torch.save(artifacts,AUDIT/'native-multi-target-artifacts.pt')


def test_v5_mode_disables_withdrawn_affine_credit():
    from reward_readout import DirectActionTargetReadout
    from types import SimpleNamespace
    readout=DirectActionTargetReadout(SimpleNamespace(),SimpleNamespace(eos_token_id=1),
        task='TextCraft',packed_answer_targets=None)
    assert os.environ['DT_CREDIT_MODE']=='reward_target_exponential_v5'
    assert readout.reward_target_exponential and not readout.weighted_target_credit
    # Actual DT answer owner is the fast original, without v4 seed transforms.
    source=(Path(os.environ['DT_ROOT'])/'clean/qwen35/qwen35_answer_finite.py').read_text()
    assert 'credit_seed_weights' not in source


def test_official_GRPO_PPO_and_fast_owner_bindings():
    import hashlib
    import importlib
    import inspect
    expected={
        'verl.trainer.ppo.core_algos':'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
        'verl.workers.actor.dp_actor':'3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
        'verl.utils.experimental.torch_functional':'bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2',
    }
    for name,digest in expected.items():
        path=Path(inspect.getsourcefile(importlib.import_module(name)))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(name,str(path))
    for name,digest in {
        'qwen35_answer_finite.py':'1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e',
        'qwen35_dense_finite_runner.py':'198f78db4abf58cd8461c5bab86d1ce9c325a46e4295c8cc54d83bd9e20e59cd',
    }.items():
        path=Path(os.environ['DT_ROOT'])/'clean/qwen35'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert not torch.cuda.is_initialized()


if __name__=='__main__':
    # Original owner transport tests are part of the same invocation, with
    # their native DataProto dedup/pad/DP/scatter implementation unchanged.
    raise SystemExit(pytest.main([__file__,str(Path(__file__).with_name('test_multi_target_v5.py')),
                                 str(ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source/test_direct_target_transport.py'),
                                 '-q','--disable-warnings','--junitxml='+str(AUDIT/'CPU-behavior-v5.xml')]))
