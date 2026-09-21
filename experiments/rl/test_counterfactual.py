"""Tests of sampled reward composition, not a DT model or task-training test."""
import math
import pytest
import torch

from counterfactual import reward_event_token_credit, assert_token_advantage_contract


def compose(d, rewards, future=None, policy=None, discounts=None):
    if future is None:
        future = torch.ones_like(d, dtype=torch.bool)
    if policy is None:
        policy = torch.ones((d.shape[0], d.shape[2]), dtype=torch.bool, device=d.device)
    return reward_event_token_credit(d, rewards, future, policy, discounts=discounts)


def test_sample_correction_is_not_raw_probability_difference():
    d = torch.tensor([[[math.log(0.6 / 0.3)]]], dtype=torch.float64)
    result = compose(d, torch.tensor([[2.0]], dtype=torch.float64))
    torch.testing.assert_close(result.advantages, torch.tensor([[1.0]], dtype=torch.float64))
    assert result.advantages.item() != pytest.approx(2.0 * (0.6 - 0.3))
    torch.testing.assert_close(result.q_estimates, torch.tensor([[2.0]], dtype=torch.float64))
    torch.testing.assert_close(result.v_estimates, torch.tensor([[1.0]], dtype=torch.float64))


def test_reward_event_expectation_equals_full_q_minus_v():
    # Four outcome trajectories of a one-box finite-horizon reward fixture.
    # Rewards are fixtures, not a local implementation of gym_sokoban.
    rewards = torch.tensor([
        [-0.1, -0.1, 10.9],
        [-0.1, 10.9, 0.0],
        [-0.1, -0.1, -0.1],
        [10.9, 0.0, 0.0],
    ], dtype=torch.float64)
    p_action = torch.tensor([
        [0.5, 0.2, 0.2, 0.1],
        [0.1, 0.4, 0.1, 0.4],
        [0.1, 0.1, 0.7, 0.1],
    ], dtype=torch.float64)
    old_policy = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    p_policy = old_policy @ p_action
    returns = rewards.sum(-1)
    q_exact = p_action @ returns
    v_exact = p_policy @ returns
    sampled_means = []
    for a in range(3):
        # Event outcome is its reward class. Sum mass over outcomes giving
        # that class; do not condition only on episodes where an event occurs.
        d = torch.empty((4, 3, 1), dtype=torch.float64)
        for outcome in range(4):
            for event in range(3):
                same_class = rewards[:, event] == rewards[outcome, event]
                pa = p_action[a, same_class].sum()
                pp = p_policy[same_class].sum()
                d[outcome, event, 0] = (pa / pp).log()
        result = compose(d, rewards)
        mean_a = p_action[a] @ result.advantages[:, 0]
        mean_q = p_action[a] @ result.q_estimates[:, 0]
        mean_v = p_action[a] @ result.v_estimates[:, 0]
        torch.testing.assert_close(mean_a, q_exact[a] - v_exact)
        torch.testing.assert_close(mean_q, q_exact[a])
        torch.testing.assert_close(mean_v, v_exact)
        sampled_means.append(mean_a)
    torch.testing.assert_close(old_policy @ torch.stack(sampled_means), torch.tensor(0.0, dtype=torch.float64), atol=1e-14, rtol=0)


def test_terminal_is_one_reward_event():
    d = torch.tensor([[[math.log(2), 0.0, math.log(0.5)]]], dtype=torch.float64)
    result = compose(d, torch.tensor([[10.0]], dtype=torch.float64))
    torch.testing.assert_close(result.advantages, torch.tensor([[5.0, 0.0, -10.0]], dtype=torch.float64))
    assert result.advantages.sum().item() != 10.0


def test_observations_and_past_rewards_are_not_routed():
    policy = torch.tensor([[True, True, False, True]])
    future = torch.tensor([[[True, True, False, False], [True, True, False, True]]])
    d = torch.tensor([[[math.log(2), math.log(0.5), float("nan"), float("nan")],
                       [0.0, math.log(2), float("nan"), math.log(4)]]], dtype=torch.float64)
    result = compose(d, torch.tensor([[-0.1, 10.9]], dtype=torch.float64), future, policy)
    expected = torch.tensor([[-0.05, 0.1 + 5.45, 0.0, 8.175]], dtype=torch.float64)
    torch.testing.assert_close(result.advantages, expected)
    torch.testing.assert_close(result.q_estimates, torch.tensor([[10.8, 10.8, 0.0, 10.9]], dtype=torch.float64))
    assert_token_advantage_contract(result.advantages, policy)


def test_discounts_and_signed_process_rewards():
    d = torch.full((1, 4, 1), math.log(2), dtype=torch.float64)
    rewards = torch.tensor([[-0.1, 1.0, -1.0, 10.0]], dtype=torch.float64)
    discounts = torch.tensor([[[1.0], [0.9], [0.81], [0.729]]], dtype=torch.float64)
    result = compose(d, rewards, discounts=discounts)
    expected_q = -0.1 + 0.9 - 0.81 + 7.29
    assert result.q_estimates.item() == pytest.approx(expected_q)
    assert result.advantages.item() == pytest.approx(0.5 * expected_q)


def test_zero_and_small_log_ratios():
    d = torch.tensor([[[0.0, 1e-10, -1e-10]]], dtype=torch.float32)
    result = compose(d, torch.tensor([[1.0]]))
    assert result.advantages[0, 0] == 0
    assert result.advantages[0, 1].item() == pytest.approx(1e-10, rel=1e-6)
    assert result.advantages[0, 2].item() == pytest.approx(-1e-10, rel=1e-6)


def test_zero_reward_does_not_need_log_ratio():
    d = torch.full((1, 1, 3), float("nan"))
    result = compose(d, torch.zeros(1, 1))
    assert torch.equal(result.advantages, torch.zeros(1, 3))


def test_estimates_are_detached():
    d = torch.ones(1, 2, 3, requires_grad=True)
    reward = torch.ones(1, 2, requires_grad=True)
    result = compose(d, reward)
    assert not any(x.requires_grad for x in (result.advantages, result.q_estimates, result.v_estimates))


def test_zero_discount_does_not_need_log_ratio():
    d = torch.full((1, 1, 2), float("nan"))
    result = compose(d, torch.ones(1, 1), discounts=torch.zeros_like(d))
    assert torch.equal(result.advantages, torch.zeros(1, 2))


@pytest.mark.parametrize("bad_d", [float("nan"), float("inf"), float("-inf")])
def test_invalid_active_log_ratio_fails(bad_d):
    with pytest.raises(ValueError, match="active event"):
        compose(torch.tensor([[[bad_d]]]), torch.ones(1, 1))


def test_no_silent_exponent_clipping():
    with pytest.raises(FloatingPointError, match="overflow"):
        compose(torch.tensor([[[-1000.0]]]), torch.ones(1, 1))


def test_no_token_broadcast_from_scalar_credit():
    with pytest.raises(ValueError, match="log_ratios"):
        reward_event_token_credit(
            torch.ones(1, 3), torch.ones(1, 1),
            torch.ones(1, 1, 3, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
        )


def test_32k_minibatch_four_tensor_boundary():
    # CPU adapter capacity check only: no claim of 32k model training.
    d = torch.full((4, 15, 32768), math.log(2))
    policy = torch.ones(4, 32768, dtype=torch.bool)
    policy[:, 1::3] = False
    result = compose(d, torch.full((4, 15), -0.1), policy=policy)
    assert result.advantages.shape == (4, 32768)
    assert_token_advantage_contract(result.advantages, policy)
    torch.testing.assert_close(result.advantages[policy], torch.full_like(result.advantages[policy], -0.75))
