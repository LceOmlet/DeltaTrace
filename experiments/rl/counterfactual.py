"""Sampled reward-event composition for the fixed DT Q/V contract.

DeltaTrace owns the event log likelihood ratios. Rollout owns rewards, event
order, discounts, and policy-token identities. This module only composes those
artifacts; it does not estimate ratios, route observations, or implement PPO.
See PLAN.md for the sampling and support conditions of the estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DTTokenCredit:
    """Detached sample estimates, not pointwise exact conditional Q and V."""
    advantages: torch.Tensor
    q_estimates: torch.Tensor
    v_estimates: torch.Tensor


@torch.no_grad()
def reward_event_token_credit(
    log_ratios: torch.Tensor,
    rewards: torch.Tensor,
    future_event_mask: torch.Tensor,
    policy_mask: torch.Tensor,
    *,
    discounts: torch.Tensor | None = None,
) -> DTTokenCredit:
    """Compose sampled terminal/process rewards into individual token advantages.

    log_ratios[b,k,i] = log P(y_k|h_i,a_i) - log P(y_k|h_i,pi_old).
    Each observed y_k is sampled after a_i under the old continuation policy.
    rewards[b,k] is its actual reward. future_event_mask[b,k,i] identifies
    events not yet settled at token i, using rollout event identities. Only
    policy_mask[b,i] positions are actions. Optional discounts[b,k,i] come
    from the same reward clock as PPO.

    Q_hat_i = sum_k discount_ki * r_k
    V_hat_i = sum_k discount_ki * r_k * exp(-d_ki)
    A_hat_i = sum_k discount_ki * r_k * (1 - exp(-d_ki)).

    Under PLAN.md's conditions E[A_hat_i|h_i,a_i] = Q_i - V_i.
    A terminal reward is one event. No cross-token normalization,
    probability-difference substitution, clipping, or learned baseline occurs.
    """
    if log_ratios.ndim != 3 or not log_ratios.is_floating_point():
        raise ValueError("log_ratios must be floating [batch, events, tokens]")
    batch, events, tokens = log_ratios.shape
    if rewards.shape != (batch, events) or not rewards.is_floating_point():
        raise ValueError("rewards must be floating [batch, events]")
    if future_event_mask.shape != log_ratios.shape or future_event_mask.dtype != torch.bool:
        raise ValueError("future_event_mask must be bool [batch, events, tokens]")
    if policy_mask.shape != (batch, tokens) or policy_mask.dtype != torch.bool:
        raise ValueError("policy_mask must be bool [batch, tokens]")
    inputs = [rewards, future_event_mask, policy_mask]
    if discounts is not None:
        if discounts.shape != log_ratios.shape or not discounts.is_floating_point():
            raise ValueError("discounts must be floating [batch, events, tokens]")
        inputs.append(discounts)
    if any(value.device != log_ratios.device for value in inputs):
        raise ValueError("all reward-event tensors must be on the same device")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("environment rewards must be finite")

    # Do not exponentiate observation/padding/past-event values. Zero-reward
    # events contribute zero and do not need a numerical likelihood ratio.
    active = future_event_mask & policy_mask[:, None, :] & (rewards[:, :, None] != 0)
    dtype = torch.promote_types(log_ratios.dtype, rewards.dtype)
    if discounts is not None:
        dtype = torch.promote_types(dtype, discounts.dtype)
        values = discounts[active]
        if not bool((torch.isfinite(values) & (values >= 0)).all()):
            raise ValueError("active discounts must be finite and nonnegative")
        active = active & (discounts > 0)
    if not bool(torch.isfinite(log_ratios[active]).all()):
        raise ValueError("active event log ratios must be finite; do not smooth away missing support")
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    d = torch.where(active, log_ratios.to(dtype), 0.0)
    weighted_reward = torch.where(active, rewards.to(dtype)[:, :, None], 0.0)
    if discounts is not None:
        weighted_reward = weighted_reward * torch.where(active, discounts.to(dtype), 0.0)

    # Stable near d=0: sampled importance correction, not R*(p_action-p_policy).
    event_advantages = weighted_reward * (-torch.expm1(-d))
    event_baselines = weighted_reward * torch.exp(-d)
    advantages = event_advantages.sum(dim=1)
    q_estimates = weighted_reward.sum(dim=1)
    v_estimates = event_baselines.sum(dim=1)
    if not all(bool(torch.isfinite(value).all()) for value in (advantages, q_estimates, v_estimates)):
        raise FloatingPointError("reward-event estimate overflow; no silent ratio clamp is applied")
    return DTTokenCredit(advantages, q_estimates, v_estimates)


def assert_token_advantage_contract(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    """Validate the existing upstream PPO tensor boundary."""
    if advantages.shape != response_mask.shape:
        raise ValueError("token advantages and response_mask must have equal shapes")
    if response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be bool")
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("token advantages must be finite")
    if bool((advantages[~response_mask].abs() > atol).any()):
        raise ValueError("token advantages must be zero outside the policy mask")
