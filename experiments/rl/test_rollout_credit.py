"""Reward-event transport through the installed VERL collector and trainer.

Ratios below are explicit test fixtures, not outputs claimed from DeltaTrace.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage
from verl.trainer.ppo.core_algos import compute_policy_loss

from counterfactual import reward_event_credit_for_episode


def episode_rows():
    # Prompt tokens represent observations. A repeated response token must
    # remain a different action at each position and in each environment step.
    rows = []
    for step, (reward, active) in enumerate([(-0.1, True), (10.9, True), (123.0, False)]):
        rows.append({
            "traj_uid": "episode", "uid": "prompt", "env_step": step,
            "rewards": reward, "active_masks": active,
            "responses": torch.tensor([7, 7, 8, 0]),
            "input_ids": torch.tensor([40, 41, 7, 7, 8, 0]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 0]),
            "dt_env_outcome": {"observation": {"text": f"state-{step}"}, "done": step >= 1},
        })
    return rows


def ratio_fixture():
    return [
        torch.tensor([[0.0, 0.0, 0.0, float("nan")],
                      [0.2, -0.3, 0.7, float("nan")]]),
        torch.tensor([[float("nan")] * 4,
                      [0.1, -0.5, 0.9, float("nan")]]),
        torch.full((2, 4), float("nan")),
    ]


def test_original_rows_keep_future_events_and_individual_tokens():
    rows = episode_rows()
    original = deepcopy(rows)
    credit = reward_event_credit_for_episode(rows, ratio_fixture())
    torch.testing.assert_close(credit[0]["dt_q_estimates"], torch.tensor([10.8, 10.8, 10.8, 0.0]))
    torch.testing.assert_close(credit[1]["dt_q_estimates"], torch.tensor([10.9, 10.9, 10.9, 0.0]))
    for name, value in credit[2].items():
        assert torch.count_nonzero(value) == 0, name
    for index in range(2):
        result = credit[index]
        assert result["dt_token_advantages"][0] != result["dt_token_advantages"][1]
        torch.testing.assert_close(result["dt_token_advantages"], result["dt_q_estimates"] - result["dt_v_estimates"])
        assert result["dt_token_advantages"][-1] == 0
    for before, after in zip(original, rows):
        torch.testing.assert_close(before["responses"], after["responses"])
        assert before["dt_env_outcome"] == after["dt_env_outcome"]


def test_collector_to_trainer_keeps_q_v_and_upstream_actor_gradient(monkeypatch):
    from verl.trainer.ppo import core_algos
    def reject_second_gae(*args, **kwargs):
        pytest.fail('DT already sums future reward events; it must not enter GAE again')
    monkeypatch.setattr(core_algos, 'compute_gae_advantage_return', reject_second_gae)
    rows = episode_rows()
    for row, values in zip(rows, reward_event_credit_for_episode(rows, ratio_fixture())):
        row.update(values)
    config = SimpleNamespace(algorithm=SimpleNamespace(adv_estimator="deltatrace"))
    collector = TrajectoryCollector(config, tokenizer=None)
    data = collector.gather_rollout_data(
        [rows], np.array([10.8]), np.array([2]), {}, np.array(["episode"]), np.array([2]),
    )
    assert len(data) == 2  # Inactive terminal padding did not become an event/action.
    original_ids = data.batch["responses"].clone()
    original_advantages = data.batch["dt_token_advantages"].clone()
    data = compute_advantage(data, AdvantageEstimator.DELTATRACE)
    torch.testing.assert_close(data.batch["advantages"], original_advantages, atol=0, rtol=0)
    torch.testing.assert_close(data.batch["responses"], original_ids)
    torch.testing.assert_close(data.batch["returns"], data.batch["dt_q_estimates"])
    assert not torch.equal(data.batch["returns"], data.batch["advantages"])
    torch.testing.assert_close(data.batch["advantages"], data.batch["dt_q_estimates"] - data.batch["dt_v_estimates"])
    assert "values" not in data.batch  # No learned critic inserted.
    old_log_prob = torch.full_like(data.batch["advantages"], -2.0)
    log_prob = old_log_prob.clone().requires_grad_()
    loss, *_ = compute_policy_loss(
        old_log_prob, log_prob, data.batch["advantages"], data.batch["response_mask"],
        cliprange=0.2, clip_ratio_c=float("inf"),
    )
    loss.backward()
    expected = -data.batch["advantages"] / data.batch["response_mask"].sum()
    torch.testing.assert_close(log_prob.grad, expected)


def test_episode_discount_input_and_terminal_event():
    rows = episode_rows()[:2]
    rows[0]["rewards"] = 0.0
    d = ratio_fixture()[:2]
    discounts = [torch.full((2, 4), 0.5), torch.ones(2, 4)]
    result = reward_event_credit_for_episode(rows, d, discounts=discounts)
    assert result[0]["dt_q_estimates"][0] == pytest.approx(10.9 * 0.5)
    assert result[1]["dt_q_estimates"][0] == pytest.approx(10.9)


@pytest.mark.parametrize("change", ["mixed_trajectory", "duplicate_step", "event_shape"])
def test_rejects_actual_event_identity_and_shape_mismatches(change):
    rows, d = episode_rows(), ratio_fixture()
    if change == "mixed_trajectory":
        rows[1]["traj_uid"] = "different"
    elif change == "duplicate_step":
        rows[1]["env_step"] = 0
    else:
        d[0] = torch.ones(1, 4)
    with pytest.raises(ValueError):
        reward_event_credit_for_episode(rows, d)


@pytest.mark.parametrize("estimator", ["deltatrace", "grpo"])
def test_native_collection_captures_outcomes_without_replacing_response_ids(estimator):
    # Test doubles supply environment transitions and generated artifacts;
    # the actual upstream rollout loop owns all sequencing and row collection.
    batch_size, prompt_width, response_width = 4, 32256, 512
    config = OmegaConf.create({
        'algorithm': {'adv_estimator': estimator},
        'env': {'max_steps': 15, 'rollout': {'n': 4}},
    })
    tokenizer = SimpleNamespace(batch_decode=lambda responses, **_: ["raw action"] * len(responses))
    collector = TrajectoryCollector(config, tokenizer)

    def preprocess(**_):
        return DataProto.from_single_dict({
            "input_ids": torch.ones(batch_size, prompt_width, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, prompt_width, dtype=torch.long),
            "position_ids": torch.arange(prompt_width).expand(batch_size, -1),
            "raw_prompt_ids": np.array(["owner prompt IDs"] * batch_size, dtype=object),
        })
    collector.preprocess_batch = preprocess

    class GeneratedArtifacts:
        world_size = 1

        def generate_sequences(self, batch):
            responses = torch.zeros(batch_size, response_width, dtype=torch.long)
            responses[:, :4] = torch.tensor([7, 7, 8, 9])
            response_mask = torch.zeros_like(responses)
            response_mask[:, :4] = 1
            return DataProto.from_single_dict({
                "responses": responses,
                "input_ids": torch.cat((batch.batch["input_ids"], responses), dim=-1),
                "attention_mask": torch.cat((batch.batch["attention_mask"], response_mask), dim=-1),
            })

    class Transitions:
        def __init__(self):
            self.step_number = 0
            self.shared_info = {"won": False, "postprocessed_action": "parsed action"}

        def observation(self):
            return {"text": [f"observation {self.step_number}"] * batch_size,
                    "image": None, "anchor": [self.shared_info] * batch_size}

        def reset(self, **_):
            return self.observation(), [self.shared_info] * batch_size

        def step(self, actions):
            self.step_number += 1
            self.shared_info["step"] = self.step_number
            actions[:] = [0] * batch_size  # Same mutation pattern as native projections.
            done = np.array([True] + [self.step_number == 2] * 3)
            return self.observation(), np.full(batch_size, -0.1), done, [self.shared_info] * batch_size

        def success_evaluator(self, **_):
            return {}

    initial = DataProto.from_single_dict({
        "input_ids": torch.ones(batch_size, 1, dtype=torch.long),
        "data_source": np.array(["interface_fixture"] * batch_size, dtype=object),
    })
    rows, rewards, lengths, *_ = collector.vanilla_multi_turn_loop(initial, GeneratedArtifacts(), Transitions())
    assert lengths.tolist() == [1.0, 2.0, 2.0, 2.0]
    assert rewards.tolist() == pytest.approx([-0.1, -0.2, -0.2, -0.2])
    for episode in rows:
        for step, row in enumerate(episode):
            assert row["env_step"] == step
            assert row["input_ids"].numel() == 32768
            assert row["responses"][:4].tolist() == [7, 7, 8, 9]
            if estimator == "deltatrace":
                # Later in-place environment changes did not corrupt history.
                assert row["dt_env_outcome"]["info"]["step"] == step + 1
                assert row["dt_env_outcome"]["observation"]["anchor"]["step"] == step + 1
            else:
                assert "dt_env_outcome" not in row
