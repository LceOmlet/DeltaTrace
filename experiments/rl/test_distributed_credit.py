"""Credit composition with actual VERL dispatch/pad/concat; no model surrogate claims."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.base.decorator import (
    dispatch_dp_compute_data_proto, collect_dp_compute_data_proto,
)
from agent_system.multi_turn_rollout.utils import to_list_of_dict

from counterfactual import return_credit_for_rows
from deltatrace_rollout import DeltaTraceRolloutProducer
from dt_training_batch import compute_training_credit, CREDIT_KEYS
from test_reward_readout import readout, BatchedRunner, row


class NativeDispatchFixture(WorkerGroup):
    def __init__(self, size):
        # Use native dispatch and collect functions; deliberately no Ray/model.
        self._workers = [None] * size
        self.calls = []

    def compute_dt_token_advantages(self, data):
        args, kwargs = dispatch_dp_compute_data_proto(self, data)
        assert kwargs == {}
        self.calls.append(args[0])
        outputs = []
        for chunk in args[0]:
            producer = DeltaTraceRolloutProducer.__new__(DeltaTraceRolloutProducer)
            def fixture(_self, episodes, returns, *, complete_returns):
                # Fixed exact ratio fixture to isolate transport/composition.
                return [return_credit_for_rows(rows,
                    [torch.tensor([.2, -.3, 0.]) for _ in rows], values)
                    for rows, values in zip(episodes, complete_returns)]
            producer.attribute_episodes = MethodType(fixture, producer)
            outputs.append(producer.attribute_prepared_batch(chunk))
        return collect_dp_compute_data_proto(self, outputs)


def training_data(rewards, order):
    count = len(rewards)
    return DataProto.from_dict(tensors=dict(
        input_ids=torch.tensor([[40, 41, 7, 8, 0]] * count),
        responses=torch.tensor([[7, 8, 0]] * count),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]] * count),
        token_level_rewards=torch.tensor([[0., sum(rewards), 0.]] * count),
    ), non_tensors=dict(
        traj_uid=np.array(['episode'] * count, dtype=object),
        env_step=np.arange(count), rewards=np.array(rewards),
        active_masks=np.ones(count, dtype=bool), episode_rewards=np.full(count, sum(rewards)),
    )).select_idxs(order)


@pytest.mark.parametrize('rewards', [[-.1] * 15, [0.] * 14 + [10.], [10.] + [0.] * 14])
def test_native_two_rank_dispatch_keeps_complete_future_returns(rewards):
    order = list(reversed(range(15))) + [0, 4, 12]
    data = training_data(rewards, order)
    single, double = NativeDispatchFixture(1), NativeDispatchFixture(2)
    expected = compute_training_credit(data, single, eos_token_id=99, pad_token_id=0)
    actual = compute_training_credit(data, double, eos_token_id=99, pad_token_id=0)
    for key in CREDIT_KEYS:
        torch.testing.assert_close(actual.batch[key], expected.batch[key], rtol=0, atol=0)
    for i, step in enumerate(order):
        assert actual.batch['dt_q_estimates'][i, 0].item() == pytest.approx(sum(rewards[step:]))
        assert actual.batch['dt_q_estimates'][i, -1] == 0
    chunks = double.calls[0]
    assert len(chunks) == 2 and len(chunks[0]) == len(chunks[1])
    assert len(chunks[0]) % 4 == 0
    assert all(torch.all(chunk.batch['dt_complete_return'] != 0) for chunk in chunks)


def test_zero_batch_does_not_enter_fsdp_collectives():
    group = NativeDispatchFixture(2)
    data = training_data([0., 0.], [1, 0, 1])
    output = compute_training_credit(data, group, eos_token_id=99, pad_token_id=0)
    assert not group.calls
    assert all(not output.batch[key].any() for key in CREDIT_KEYS)


@pytest.mark.parametrize('reward', [0., 1.])
def test_training_slices_do_not_drop_later_executed_reward(reward):
    # The final action is an executed environment event even when its tokens
    # are outside the author's training truncation. Keep the original reward
    # there; DT requests cover only retained training-token source identities.
    order = [2, 0, 1, 0]
    data = training_data([0., 0., reward], order)
    from dt_training_batch import prepare_training_credit
    prepared, inverse = prepare_training_credit(data)
    assert prepared.batch['dt_complete_return'].tolist() == [reward] * 3
    group = NativeDispatchFixture(2)
    actual = compute_training_credit(data, group, eos_token_id=99, pad_token_id=0,
                                     source_indices=torch.tensor([1, 2, 3]))
    if reward:
        assert len(group.calls) == 1
        # Native B8 transport padding can duplicate eligible requests, but
        # cannot cause the excluded terminal response to enter a model call.
        for chunk in group.calls[0]:
            assert set(chunk.non_tensor_batch['env_step']) <= {0, 1}
            assert chunk.batch['dt_complete_return'].tolist() == [reward] * len(chunk)
    else:
        assert not group.calls
    for index, step in enumerate(order):
        expected = 0. if step == 2 else reward
        assert actual.batch['dt_q_estimates'][index, 0].item() == expected
    # Reward location and complete native response artifacts are unchanged.
    assert data.non_tensor_batch['rewards'].tolist() == [reward, 0., 0., 0.]


def test_empty_training_slice_selection_avoids_all_dt_calls():
    data = training_data([0., 1.], [0, 1])
    group = NativeDispatchFixture(2)
    actual = compute_training_credit(data, group, eos_token_id=99, pad_token_id=0,
                                     source_indices=torch.tensor([], dtype=torch.long))
    assert not group.calls
    assert all(not actual.batch[key].any() for key in CREDIT_KEYS)


def test_trajectory_uses_native_slice_sources_after_complete_returns():
    from owner_trajectory_batch import trajectory_credit
    source = training_data([0., 0., 1.], [0, 1, 2])
    # Observation/unused slots receive no policy credit. A partially retained
    # response keeps its original full source endpoint and exact slice length.
    mapping = np.empty(1, dtype=object)
    mapping[0] = [(0, 1, 1), (1, 3, 2)]
    destination = DataProto.from_dict(tensors=dict(responses=torch.tensor([[90, 7, 91, 7, 8]])),
        non_tensors=dict(dt_response_slices=mapping))
    group = NativeDispatchFixture(2)
    output = trajectory_credit(destination, source, group, eos_token_id=99, pad_token_id=0)
    assert len(group.calls) == 1
    for chunk in group.calls[0]:
        assert set(chunk.non_tensor_batch['env_step']) <= {0, 1}
        assert chunk.batch['responses'].tolist() == [[7, 8, 0]] * len(chunk)
        assert chunk.batch['dt_complete_return'].tolist() == [1.] * len(chunk)
    assert output.batch['dt_q_estimates'].tolist() == [[0., 1., 0., 1., 1.]]
    for key in CREDIT_KEYS:
        assert not output.batch[key][0, [0, 2]].any()


@pytest.mark.parametrize('count', [17, 22])
def test_owner_balance_restores_credit_after_padding_and_duplicate_rows(count, monkeypatch):
    from verl.utils import seqlen_balancing

    calls = []
    original = seqlen_balancing.get_seqlen_balanced_partitions
    def observe(lengths, *, k_partitions, equal_size):
        partitions = original(lengths, k_partitions, equal_size)
        calls.append((lengths, partitions))
        return partitions
    monkeypatch.setattr(seqlen_balancing, 'get_seqlen_balanced_partitions', observe)
    order = list(reversed(range(count))) + [0, count - 1, 3]
    data = training_data([.1] * count, order)
    lengths = torch.tensor([8 + step * 4 for step in order])
    width = int(lengths.max()) + 3
    mask = torch.arange(width - 3)[None, :] >= (width - lengths[:, None])
    data.batch['attention_mask'] = torch.cat((mask.long(), data.batch['attention_mask'][:, -3:]), -1)
    data.batch['input_ids'] = torch.cat((torch.full_like(mask, 40, dtype=torch.long), data.batch['responses']), -1)
    expected = compute_training_credit(data, NativeDispatchFixture(1), eos_token_id=99, pad_token_id=0)
    group = NativeDispatchFixture(2)
    actual = compute_training_credit(data, group, eos_token_id=99, pad_token_id=0)
    for key in CREDIT_KEYS:
        torch.testing.assert_close(actual.batch[key], expected.batch[key], rtol=0, atol=0)
    original_lengths, partitions = calls[-1]
    assert len(original_lengths) % 8 == 0
    chunks = group.calls[0]
    assert [chunk.batch['attention_mask'].sum(-1).tolist() for chunk in chunks] == [
        [original_lengths[index] for index in partition] for partition in partitions]
    assert all(len(chunk) % 4 == 0 for chunk in chunks)
    # The old contiguous split reproduced the observed short-rank/long-rank skew.
    midpoint = len(original_lengths) // 2
    old_spread = abs(sum(original_lengths[:midpoint]) - sum(original_lengths[midpoint:]))
    new_spread = abs(sum(original_lengths[i] for i in partitions[0]) -
                     sum(original_lengths[i] for i in partitions[1]))
    assert new_spread < old_spread


def test_native_denominators_stay_collectively_aligned_without_changing_credit():
    data = training_data([0., 0., 1.], [2, 0, 1, 2])
    data.non_tensor_batch['appworld_num_tests'] = np.array([2, 3, 4, 2])
    group = NativeDispatchFixture(2)
    actual = compute_training_credit(data, group, eos_token_id=99, pad_token_id=0)
    assert len(group.calls) == 3
    for chunks in group.calls:
        assert len(set(int(n) for chunk in chunks for n in chunk.non_tensor_batch['appworld_num_tests'])) == 1
    torch.testing.assert_close(actual.batch['dt_q_estimates'][:, 0], torch.ones(4))


def test_prepared_readout_does_not_recompute_return_on_a_partial_episode():
    # These rows are independent current responses; their later rewards live
    # on other ranks. The explicit complete return must select the target and
    # feed Q/V, rather than recomputing future rewards from this partial list.
    rows = [row(0, -.1), row(3, -.1), row(7, -.1), row(13, -.1)]
    values = [9.5, 9.8, 10.2, 10.8]
    dt = readout(BatchedRunner(), minibatch_size=4)
    result = dt.episodes([rows], complete_returns=[values])[0]
    assert dt.last_report['finite_trace_calls'] == 1
    for output, value in zip(result, values):
        assert output['dt_q_estimates'][0].item() == pytest.approx(value)
    assert [t['observed_return'] for t in dt.last_report['traces']] == values


def test_standalone_capacity_fixture_uses_current_native_rpc():
    from verify_dt_context_capacity import attribute_saved_episodes
    episodes = [[row(0, -.1), row(1, 10.9)], [row(0, -.1)]]
    seen = []
    def original_interface(data):
        seen.append(data)
        return DataProto.from_dict(tensors={key: data.batch['dt_complete_return'][:, None].float()
            for key in CREDIT_KEYS})
    worker = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=99, pad_token_id=0),
        compute_dt_token_advantages=original_interface)
    output = attribute_saved_episodes(worker, episodes)
    assert [len(values) for values in output] == [2, 1]
    assert seen[0].batch['dt_complete_return'].tolist() == pytest.approx([10.8, 10.9, -.1])
    assert seen[0].meta_info == dict(eos_token_id=99, pad_token_id=0)
    torch.testing.assert_close(seen[0].batch['input_ids'][0], episodes[0][0]['input_ids'])
