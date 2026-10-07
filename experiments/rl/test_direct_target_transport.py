"""CPU transport tests using fixed VERL containers, padding and DP dispatch.

The worker below returns token identities solely to test transport/scatter.
It is not a DT estimator, numerical reference, model or environment parser.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.base.decorator import (
    dispatch_dp_compute_data_proto, collect_dp_compute_data_proto,
)
from agent_system.multi_turn_rollout.utils import adjust_batch

from dt_training_batch import compute_direct_target_credit, CREDIT_KEYS


class IdentityTransportWorker(WorkerGroup):
    """Use original VERL dispatch/collect with explicitly non-numerical outputs."""

    def __init__(self, size=2):
        self._workers = [None] * size
        self.calls = []

    def compute_dt_token_advantages(self, data):
        args, kwargs = dispatch_dp_compute_data_proto(self, data)
        assert kwargs == {}
        chunks = args[0]
        self.calls.append(chunks)
        outputs = []
        for chunk in chunks:
            assert len(chunk) % 4 == 0
            assert chunk.meta_info['dt_target_semantics'] == 'native_joint_action_target'
            # Each genuine target set remains one row and carries one reward.
            assert torch.all(chunk.batch['dt_direct_reward'] == .75)
            assert torch.all(chunk.batch['target_mask'].sum(-1) == 2)
            values = chunk.batch['responses'].float() * chunk.batch['policy_mask']
            outputs.append(DataProto.from_dict(tensors={
                key: values + 1000 * k * chunk.batch['policy_mask']
                for k, key in enumerate(CREDIT_KEYS)
            }))
        return collect_dp_compute_data_proto(self, outputs)


def artifact(index, *, policy=None, target=None, retained=None):
    # Interspersed observation token at slot 2 and multiple executed targets.
    ids = [100 + index * 10 + j for j in range(6)]
    return dict(prompt_ids=[10 + index] * (index + 1), response_ids=ids,
                policy_mask=policy or [True, True, False, True, True, True],
                target_mask=target or [False, True, False, False, True, False],
                retained_response_positions=retained or [0, 1, 2, 3])


def native_training_data(artifacts, *, rewards=None, identities=None):
    count = len(artifacts)
    tensors = []
    for item in artifacts:
        tensors.append([item['response_ids'][position] if position >= 0 else 0
                        for position in item['retained_response_positions']])
    responses = torch.tensor(tensors, dtype=torch.long)
    scores = torch.zeros_like(responses, dtype=torch.float32)
    scores[:, -1] = torch.tensor(rewards if rewards is not None else [.75] * count)
    objects = np.empty(count, dtype=object)
    objects[:] = artifacts
    return DataProto.from_dict(tensors=dict(
        input_ids=torch.cat((torch.full((count, 1), 9, dtype=torch.long), responses), -1),
        responses=responses, token_level_rewards=scores,
    ), non_tensors=dict(
        traj_uid=np.array(identities or [f'uid-{i}' for i in range(count)], dtype=object),
        dt_direct_target_artifact=objects,
    ))


def test_native_adjust_pad_dispatch_deduplicates_and_restores_retained_slots():
    originals = [artifact(i) for i in range(5)]
    data = native_training_data(originals)
    config = SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=2, nnodes=1),
        algorithm=SimpleNamespace(use_kl_in_reward=False),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=4),
            actor=SimpleNamespace(use_kl_loss=False, ppo_micro_batch_size_per_gpu=4)))
    # This is the owner's actual copy-to-divisor operation, not a local clone.
    data = adjust_batch(config, data)
    assert len(data) == 8 and len(set(data.non_tensor_batch['traj_uid'])) == 5
    # A duplicated training row may retain different original suffix slots.
    row = 5
    item = deepcopy(data.non_tensor_batch['dt_direct_target_artifact'][row])
    item['retained_response_positions'] = [3, 4, 5, -1]
    data.non_tensor_batch['dt_direct_target_artifact'][row] = item
    data.batch['responses'][row] = torch.tensor([
        item['response_ids'][position] if position >= 0 else 0
        for position in item['retained_response_positions']])
    group = IdentityTransportWorker()
    output = compute_direct_target_credit(data, group, eos_token_id=99, pad_token_id=0)
    assert len(group.calls) == 1
    chunks = group.calls[0]
    assert [len(chunk) for chunk in chunks] == [4, 4]
    assert set(uid for chunk in chunks for uid in chunk.non_tensor_batch['traj_uid']) == {
        f'uid-{i}' for i in range(5)}
    for chunk in chunks:
        for i, identity in enumerate(chunk.non_tensor_batch['traj_uid']):
            original = originals[int(identity.split('-')[-1])]
            actual = chunk.batch['input_ids'][i][chunk.batch['attention_mask'][i].bool()]
            assert actual.tolist() == original['prompt_ids'] + original['response_ids']
            assert chunk.batch['policy_mask'][i].tolist() == original['policy_mask']
            assert chunk.batch['target_mask'][i].tolist() == original['target_mask']
            assert not chunk.batch['target_mask'][i, 2]  # O remains state.
    for i, item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        for slot, position in enumerate(item['retained_response_positions']):
            for k, key in enumerate(CREDIT_KEYS):
                expected = (item['response_ids'][position] + 1000*k
                            if position >= 0 and item['policy_mask'][position] else 0.)
                assert output.batch[key][i, slot].item() == expected


def test_trace_required_excludes_analytic_cases_before_native_collective():
    items = [artifact(0), artifact(1), artifact(2), artifact(3)]
    items[1]['target_mask'] = [False] * 6  # Empty literal event; original r retained.
    items[2]['policy_mask'] = [False, True, False, False, True, True]
    items[2]['target_mask'] = [False, True, False, False, True, False]
    # For row 2, the only non-target policy token is after the last target.
    data = native_training_data(items, rewards=[.75, .75, .75, 0.])
    group = IdentityTransportWorker()
    result = compute_direct_target_credit(data, group, eos_token_id=99, pad_token_id=0)
    assert len(group.calls) == 1
    assert {uid for c in group.calls[0] for uid in c.non_tensor_batch['traj_uid']} == {'uid-0'}
    # Native padding is B8 despite just one genuine DT request.
    assert sum(len(c) for c in group.calls[0]) == 8
    assert result.batch['dt_token_advantages'][1].tolist() == [0., 0., 0., 0.]
    assert result.batch['dt_q_estimates'][1].tolist() == [.75, .75, 0., .75]
    assert result.batch['dt_v_estimates'][1].tolist() == [.75, .75, 0., .75]
    assert result.batch['dt_token_advantages'][2].tolist() == [0., .75, 0., 0.]
    assert not any(result.batch[key][3].any() for key in CREDIT_KEYS)


def test_one_terminal_reward_is_not_repeated_for_multiple_target_spans():
    item = artifact(0, policy=[False, True, False, False, True, True])
    item['retained_response_positions'] = [1, 4, 5, -1]
    data = native_training_data([item], rewards=[.75])
    group = IdentityTransportWorker()
    result = compute_direct_target_credit(data, group, eos_token_id=99, pad_token_id=0)
    assert not group.calls  # self targets and causally later source are analytic.
    assert result.batch['dt_q_estimates'].tolist() == [[.75, .75, .75, 0.]]
    assert result.batch['dt_v_estimates'].tolist() == [[0., 0., .75, 0.]]
    assert result.batch['dt_token_advantages'].tolist() == [[.75, .75, 0., 0.]]


def test_observation_cannot_be_an_executed_action_target():
    item = artifact(0)
    item['target_mask'][2] = True
    with pytest.raises(ValueError, match='policy tokens'):
        compute_direct_target_credit(native_training_data([item]), IdentityTransportWorker(),
                                     eos_token_id=99, pad_token_id=0)


def test_retained_transport_rejects_changed_native_token_identity():
    item = artifact(0)
    item['target_mask'] = [False] * 6
    data = native_training_data([item])
    data.batch['responses'][0, 1] += 1
    with pytest.raises(ValueError, match='token identity'):
        compute_direct_target_credit(data, IdentityTransportWorker(),
                                     eos_token_id=99, pad_token_id=0)
