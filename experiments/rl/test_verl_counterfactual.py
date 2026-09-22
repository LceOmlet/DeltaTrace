"""Actual pinned VERL policy-loss integration; no copied PPO implementation."""
import math

import pytest
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss

from counterfactual import reward_event_token_credit


def test_exact_eos_credit_has_same_expected_actor_gradient_for_common_deletion_baseline():
    # Exact finite event laws, explicitly fixtures rather than a DT predictor.
    # The common deletion distribution differs from the policy marginal.
    logits = torch.tensor([-.4, .2, .7], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    policy = logp.detach().exp()
    laws = torch.tensor([[.7, .2, .1], [.2, .3, .5], [.4, .4, .2]], dtype=torch.float64)
    deleted = torch.tensor([.25, .3, .45], dtype=torch.float64)
    rewards = torch.tensor([-.1, 0., 10.9], dtype=torch.float64)
    q = laws @ rewards
    v, b = policy @ q, deleted @ rewards
    assert abs(float(v-b)) > .1  # Does not assume EOS value equals PPO V.
    dt_loss, exact_loss = logits.new_zeros(()), logits.new_zeros(())
    mask = torch.ones(1, 1, dtype=torch.bool)
    for action in range(3):
        d = (laws[action].log()-deleted.log()).reshape(3, 1, 1)
        credit = reward_event_token_credit(d, rewards[:, None],
                    torch.ones_like(d, dtype=torch.bool), torch.ones(3, 1, dtype=torch.bool))
        torch.testing.assert_close(laws[action] @ credit.advantages[:, 0], q[action]-b)
        for outcome in range(3):
            loss, *_ = compute_policy_loss(
                logp[action].detach().reshape(1, 1), logp[action].reshape(1, 1),
                credit.advantages[outcome:outcome+1], mask,
                cliprange=.2, clip_ratio_c=float('inf'),
            )
            dt_loss = dt_loss + policy[action] * laws[action, outcome] * loss
        loss, *_ = compute_policy_loss(
            logp[action].detach().reshape(1, 1), logp[action].reshape(1, 1),
            (q[action]-v).reshape(1, 1), mask, cliprange=.2, clip_ratio_c=float('inf'),
        )
        exact_loss = exact_loss + policy[action] * loss
    dt_gradient = torch.autograd.grad(dt_loss, logits, retain_graph=True)[0]
    exact_gradient = torch.autograd.grad(exact_loss, logits)[0]
    torch.testing.assert_close(dt_gradient, exact_gradient, atol=1e-12, rtol=1e-12)


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
