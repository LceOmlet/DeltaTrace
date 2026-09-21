"""Actual pinned VERL policy-loss integration; no copied PPO implementation."""
import math

import pytest
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss

from counterfactual import reward_event_token_credit


@pytest.mark.parametrize("ratios,expected_loss,expected_gradient", [
    ([1.0, 1.0, 1.0, 1.0], 0.0, [-1/3, 2/3, 0.0, -1/3]),
    ([1.5, 0.5, 5.0, 1.0], -0.2, [0.0, 0.0, 0.0, -1/3]),
    ([1.0, 4.0, 5.0, 1.0], 2.0, [-1/3, 8/3, 0.0, -1/3]),
])
def test_sampled_advantages_enter_upstream_ppo(ratios, expected_loss, expected_gradient):
    policy = torch.tensor([[True, True, False, True]])
    future = torch.ones(1, 1, 4, dtype=torch.bool)
    d = torch.tensor([[[math.log(2), math.log(0.5), float("nan"), math.log(2)]]])
    credit = reward_event_token_credit(d, torch.tensor([[2.0]]), future, policy)
    # The observation has no actor gradient. Clipped positions use the
    # upstream implementation, with dual clipping explicitly disabled.
    old_log_prob = torch.full((1, 4), -3.0)
    log_prob = (old_log_prob + torch.tensor([ratios]).log()).requires_grad_()
    loss, *_ = compute_policy_loss(
        old_log_prob, log_prob, credit.advantages, policy,
        cliprange=0.2, clip_ratio_c=float("inf"),
    )
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss, abs=1e-6)
    torch.testing.assert_close(log_prob.grad, torch.tensor([expected_gradient]))
