"""v11's only new behavior: native GRPO broadcast on target-free rollouts."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'candidate-entry'))
sys.path.insert(1, os.environ['DT_VERL_ROOT'])
sys.path.append(str(Path(os.environ['DT_ROOT'])/'clean/qwen35'))
from native_contract_fixtures import make_data, PrimitiveContractWorker
from verl import DataProto
from saved_native_outputs import SavedNativeOutputs, PAIR, digest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load('original_v5_batch', HERE/'baseline-entry/dt_training_batch.py')
NEW = load('v11_native_batch', HERE/'candidate-entry/dt_training_batch.py')


def compute(module, data, worker=None):
    return module.compute_direct_target_credit(data, worker or PrimitiveContractWorker(),
        eos_token_id=248046, pad_token_id=0)


def test_v5_default_is_bitwise_unchanged(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v5')
    data = make_data(duplicate=True)
    before, after = compute(BASE,data), compute(NEW,data)
    assert all(torch.equal(before.batch[k],after.batch[k]) for k in BASE.CREDIT_KEYS)


def test_v11_empty_target_and_target_paths(monkeypatch):
    data = make_data(duplicate=True)
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v5')
    before = compute(BASE,data)
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    after = compute(NEW,data)
    expected = data.batch['dt_grpo_multipliers'][0]*data.batch['response_mask'][0]
    assert torch.equal(after.batch['dt_token_advantages'][0],expected)
    assert torch.equal(after.batch['dt_q_estimates'][0],expected)
    assert not bool(after.batch['dt_v_estimates'][0].any())
    for k in BASE.CREDIT_KEYS:
        assert torch.equal(before.batch[k][1:],after.batch[k][1:])


def test_all_empty_batch_requires_no_DT_RPC_and_keeps_zero_g(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    data = make_data()
    for item in data.non_tensor_batch['dt_direct_target_artifact']:
        item['target_mask'] = [False]*len(item['target_mask'])
    data.batch['dt_grpo_multipliers'][2] = 0
    class NoDT:
        def compute_dt_token_advantages(self, *_):
            raise AssertionError('No DT call is permitted for empty targets')
    after = compute(NEW,data,NoDT())
    expected = data.batch['dt_grpo_multipliers'][:,None]*data.batch['response_mask']
    assert torch.equal(after.batch['dt_token_advantages'],expected)
    assert torch.equal(after.batch['dt_q_estimates'],expected)
    assert not bool(after.batch['dt_v_estimates'].any())


def test_future_observation_and_padding_masks_are_preserved(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    data = make_data()
    data.non_tensor_batch['dt_direct_target_artifact'][0]['future_policy_mask'] = [True,True,False,True,True,True,True,True,True]
    after = compute(NEW,data)
    assert after.batch['dt_token_advantages'][0,2] == 0
    for k in BASE.CREDIT_KEYS:
        assert not bool(after.batch[k][~data.batch['response_mask'].bool()].any())


def test_single_explicit_event_matches_native_single_target(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    data = make_data()
    single = compute(NEW,data)
    for i,item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        item['dt_reward_target_events'] = [dict(event_id=0,
            target_offsets=[p for p,t in enumerate(item['target_mask']) if t],
            credit=float(data.batch['dt_grpo_multipliers'][i]), future_policy_mask=[True]*9)]
    multi = compute(NEW,data)
    assert all(torch.equal(single.batch[k],multi.batch[k]) for k in BASE.CREDIT_KEYS)


def test_empty_component_of_targeted_rollout_is_not_a_GRPO_rollout_fallback(monkeypatch):
    data = make_data().select_idxs([1])
    g = float(data.batch['dt_grpo_multipliers'][0])
    item = data.non_tensor_batch['dt_direct_target_artifact'][0]
    item['dt_reward_target_events'] = [
        dict(event_id='actual',target_offsets=[7],credit=g/2,future_policy_mask=[True]*9),
        dict(event_id='empty',target_offsets=[],credit=g/2,future_policy_mask=[True]*9)]
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v5')
    before = compute(BASE,data)
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    after = compute(NEW,data)
    assert all(torch.equal(before.batch[k],after.batch[k]) for k in BASE.CREDIT_KEYS)


def test_readout_uses_original_v5_exponential_and_constant_self_target(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    module = load('v11_readout_owner',HERE/'candidate-entry/reward_readout.py')
    runner = SimpleNamespace(model=SimpleNamespace(execution_device=torch.device('cpu'),
        lm_head=SimpleNamespace(weight=torch.empty(1))))
    reader = module.DirectActionTargetReadout(runner,SimpleNamespace(eos_token_id=0),
        task='TextCraft',packed_answer_targets=object,minibatch_size=4)
    assert reader.reward_target_exponential
    assert not reader.target_token_logprob
    assert not reader.weighted_target_credit


def test_saved_natural_DT_outputs_through_original_DataProto_and_scatter(monkeypatch):
    replay = SavedNativeOutputs()
    paths = [PAIR/f/'002-update_actor-input.pkl' for f in ['rank0-pid1021020','rank1-pid1023296']]
    data = DataProto.concat([DataProto.load_from_disk(str(p)) for p in paths])
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v5')
    before = compute(BASE,data,replay)
    default = compute(NEW,data,replay)
    assert all(torch.equal(before.batch[k],default.batch[k]) for k in BASE.CREDIT_KEYS)
    monkeypatch.setenv('DT_CREDIT_MODE','reward_target_exponential_v11')
    after = compute(NEW,data,replay)
    targeted = torch.tensor([any(item['target_mask']) for item in
                            data.non_tensor_batch['dt_direct_target_artifact']])
    empty = ~targeted
    mask = data.batch['response_mask'].bool()
    expected = data.batch['dt_grpo_multipliers'][:,None]*mask
    for k in BASE.CREDIT_KEYS:
        assert torch.equal(before.batch[k][targeted],after.batch[k][targeted])
        assert not bool(after.batch[k][~mask].any())
        assert after.batch[k].isfinite().all()
    assert torch.equal(after.batch['dt_token_advantages'][empty],expected[empty])
    assert torch.equal(after.batch['dt_q_estimates'][empty],expected[empty])
    assert not bool(after.batch['dt_v_estimates'][empty].any())
    record = dict(status='passed',scope='Saved original DP2 natural iteration6, not a fresh DP8 model test',
        rows=len(data),has_target=int(targeted.sum()),empty_target=int(empty.sum()),
        nonzero_g_empty_target=int((empty & (data.batch['dt_grpo_multipliers'] != 0)).sum()),
        default_v5_bitwise_unchanged=True,targeted_raw_QVA_bitwise_unchanged=True,
        source_mean_adjustment=False,thresholds_used=False,new_model_DT_optimizer_environment_calls=0,
        sources=replay.sources+[digest(p) for p in paths])
    (HERE.parent/'saved-native-fallback-verification.json').write_text(json.dumps(record,indent=2)+'\n')
