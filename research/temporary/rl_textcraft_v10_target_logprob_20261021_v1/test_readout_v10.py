"""Native readout representation/default-path checks; model checks run separately."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
ENTRY=ROOT/'candidates/textcraft-v10-target-logprob-20261021-v1/entry'
ORIGINAL=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/entry'
sys.path.append(str(Path(os.environ['DT_ROOT'])/'clean/qwen35'))


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def rows():
    result=[]
    for i,(target,credit) in enumerate([([3,5],1.),([2,4,5],-.5),([3,5],0.),([0,2,3,5],.75)]):
        response=torch.arange(10+8*i,18+8*i)
        attention=torch.tensor([0,1,1,1,1,1,1,1,1,0,0],dtype=torch.bool)
        policy=torch.tensor([1,0,1,1,0,1,0,0],dtype=torch.bool)
        mask=torch.zeros(8,dtype=torch.bool);mask[target]=True
        policy |= mask
        future=torch.ones(8,dtype=torch.bool)
        if i==1:future[5]=False
        result.append(dict(input_ids=torch.cat((torch.tensor([0,8,9]),response)),
            attention_mask=attention,responses=response,policy_mask=policy,target_mask=mask,
            dt_future_event_mask=future,dt_direct_reward=credit,traj_uid=f'actual-row-{i}'))
    return result


def exercise(module, mode, *, missing_scores=False, native_dtype=torch.float32):
    os.environ['DT_CREDIT_MODE']=mode
    calls=[]
    def owner_contract(runner,reference,factual,cases,offsets,**kwargs):
        from qwen35_answer_finite import PackedAnswerTargets
        selection=PackedAnswerTargets(cases,offsets,factual.shape[1],torch.device('cpu'))
        logp0=torch.full((len(selection.labels),),-2.,dtype=native_dtype)
        delta=torch.tensor([-.4,.25,0.,-.2,.6],dtype=native_dtype)[:len(selection.labels)]
        logp1=logp0+delta
        signed=torch.zeros_like(factual,dtype=native_dtype)
        for b in range(len(cases)):
            positions=(reference[b]!=factual[b]).nonzero().flatten()
            signed[b,positions]=torch.linspace(-.125,.375,len(positions),dtype=native_dtype)
        detail=dict(target_logp0=logp0,target_logp1=logp1,
            per_sample=[dict(root_effect=float(delta[selection.samples==b].sum()),
                policy_credit_signed_sum=float(signed[b].sum()),conservation_residual=0.,
                factual_target_logp=float(logp1[selection.samples==b].sum()),
                reference_target_logp=float(logp0[selection.samples==b].sum())) for b in range(len(cases))])
        calls.append(dict(reference=reference.clone(),factual=factual.clone(),offsets=copy.deepcopy(offsets),
            signed=signed.clone(),selection=selection,logp0=logp0.clone(),logp1=logp1.clone()))
        if missing_scores:
            detail.pop('target_logp0');detail.pop('target_logp1')
        return signed,None,detail
    module.trace_token_attribution=owner_contract
    runner=types.SimpleNamespace(model=types.SimpleNamespace(execution_device=torch.device('cpu'),
        lm_head=torch.nn.Linear(1,1,bias=False)))
    readout=module.DirectActionTargetReadout(runner,types.SimpleNamespace(eos_token_id=0),
        task='TextCraft',packed_answer_targets=object,minibatch_size=4)
    inputs=rows()
    return readout,inputs,readout.trajectories(inputs),calls


def test_default_v5_remains_bitwise_identical():
    baseline=load('original_v5_readout_default',ORIGINAL/'reward_readout.py')
    candidate=load('v10_readout_default',ENTRY/'reward_readout.py')
    _,_,old,old_calls=exercise(baseline,'reward_target_exponential_v5')
    _,_,new,new_calls=exercise(candidate,'reward_target_exponential_v5')
    assert len(old_calls)==len(new_calls)==1
    for before,after in zip(old,new):
        assert all(torch.equal(before[k],after[k]) for k in before)
    for before,after in zip(old_calls,new_calls):
        assert torch.equal(before['reference'],after['reference'])
        assert torch.equal(before['factual'],after['factual'])


@pytest.mark.parametrize('native_dtype',[torch.float32,torch.float64])
def test_target_scores_keep_native_row_and_token_identity(native_dtype):
    module=load('v10_readout_identity',ENTRY/'reward_readout.py')
    readout,inputs,output,calls=exercise(module,'reward_target_logprob_exponential_v10',native_dtype=native_dtype)
    assert len(calls)==1
    call=calls[0];selection=call['selection']
    for i in (0,1):
        row=inputs[i];prepared=readout._prepare_row(row,i)
        packed=(selection.samples==i)
        d=call['logp1'][packed]-call['logp0'][packed]
        target_positions=prepared['suffix_positions'][prepared['valid_target']]
        expected=readout._prepare_row(row,i)['ratios']
        prior=prepared['prior'][prepared['suffix_positions']]
        expected[prepared['suffix_positions'][prior]]=call['signed'][i,prepared['prompt_length']:prepared['selected'].numel()][prior].to(dtype=expected.dtype)
        expected[target_positions]=d.to(dtype=expected.dtype)
        expected_credit=module.reward_event_token_credit(expected[None,None,:],
            torch.tensor([[row['dt_direct_reward']]]),row['dt_future_event_mask'][None,None,:],row['policy_mask'][None,:])
        for key,value in [('dt_q_estimates',expected_credit.q_estimates),
                          ('dt_v_estimates',expected_credit.v_estimates),
                          ('dt_token_advantages',expected_credit.advantages)]:
            assert torch.equal(output[i][key],value[0])
        # Original target labels/prefix tokens and all observations remain factual.
        assert torch.equal(call['reference'][i,prepared['prompt_length']+torch.tensor(prepared['target_offsets'])],
                           call['factual'][i,prepared['prompt_length']+torch.tensor(prepared['target_offsets'])])
    assert readout.last_report['target_self_constant_override'] is False
    assert readout.last_report['additional_model_forwards']==0


def test_source_credit_and_native_call_count_unchanged():
    module=load('v10_readout_source',ENTRY/'reward_readout.py')
    _,inputs,old,old_calls=exercise(module,'reward_target_exponential_v5')
    _,_,new,new_calls=exercise(module,'reward_target_logprob_exponential_v10')
    assert len(old_calls)==len(new_calls)==1
    for row,before,after in zip(inputs,old,new):
        source=row['policy_mask']&~row['target_mask']
        for key in before:
            assert torch.equal(before[key][source],after[key][source])


def test_zero_coefficient_equal_endpoints_and_masks():
    module=load('v10_readout_masks',ENTRY/'reward_readout.py')
    _,inputs,output,calls=exercise(module,'reward_target_logprob_exponential_v10')
    assert len(calls)==1  # Zero coefficient and no prior source need no extra DT.
    for row,result in zip(inputs,output):
        inactive=~row['policy_mask']|~row['dt_future_event_mask']
        assert all(not bool(t[inactive].any()) for t in result.values())
    assert all(not bool(t.any()) for t in output[2].values())
    assert not bool(output[3]['dt_token_advantages'].any())
    assert torch.equal(output[3]['dt_q_estimates'],output[3]['dt_v_estimates'])


def test_v10_missing_native_endpoint_scores_fails_instead_of_filling():
    module=load('v10_readout_missing',ENTRY/'reward_readout.py')
    with pytest.raises(KeyError):
        exercise(module,'reward_target_logprob_exponential_v10',missing_scores=True)
    # The pre-existing v5 owner contract does not require the new score seam.
    exercise(module,'reward_target_exponential_v5',missing_scores=True)
