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
