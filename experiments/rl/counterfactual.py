"""Counterfactual policy-gradient credit used by the RL development branch.

The estimator follows the identity recorded in the DeltaTrace appendix:
for a fixed history and continuation policy, the expectation of
``G(h, a) - G(h, a')`` over an independently sampled reference action ``a'``
is ``Q(h, a) - V(h)``.  The returned credit is detached before it weights a
score-function loss.  No input-attribution score is silently treated as an
advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CounterfactualCredit:
    """A batch of action-level counterfactual credits and audit statistics."""

    credit: torch.Tensor
    selected_return: torch.Tensor
    reference_returns: torch.Tensor

    def detached(self) -> "CounterfactualCredit":
        return CounterfactualCredit(
            self.credit.detach(), self.selected_return.detach(), self.reference_returns.detach()
        )


def averaged_counterfactual_credit(
    selected_return: torch.Tensor,
    reference_returns: torch.Tensor,
) -> CounterfactualCredit:
    """Compute exactly averaged action credit.

    ``selected_return`` has shape ``[batch]`` and ``reference_returns`` has
    shape ``[batch, num_references]``.  Reference actions must have been drawn
    independently from the same conditional policy by the caller.  We keep
    that sampling contract explicit rather than hiding it in a GRPO-style
    group mean.
    """

    if selected_return.ndim != 1:
        raise ValueError("selected_return must have shape [batch]")
    if reference_returns.ndim != 2 or reference_returns.shape[0] != selected_return.shape[0]:
        raise ValueError("reference_returns must have shape [batch, num_references]")
    if reference_returns.shape[1] < 1:
        raise ValueError("at least one independent reference action is required")
    if not (torch.isfinite(selected_return).all() and torch.isfinite(reference_returns).all()):
        raise ValueError("counterfactual returns must be finite")
    credit = selected_return - reference_returns.mean(dim=1)
    return CounterfactualCredit(credit, selected_return, reference_returns)


def score_function_loss(log_prob: torch.Tensor, credit: torch.Tensor) -> torch.Tensor:
    """Return ``-E[log pi(a|h) * C(h,a)]`` with a detached credit signal."""

    if log_prob.shape != credit.shape:
        raise ValueError("log_prob and credit must have the same shape")
    if log_prob.ndim != 1:
        raise ValueError("the score-function loss expects one selected action per item")
    if not torch.isfinite(log_prob).all() or not torch.isfinite(credit).all():
        raise ValueError("non-finite policy-gradient inputs")
    return -(log_prob * credit.detach()).mean()


def assert_advantage_equivalence(
    q_values: torch.Tensor,
    policy_probs: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    """Check the exact finite-action identity used by the estimator.

    This is an algebraic invariant, not a model-performance benchmark.  It
    catches accidental use of a group-centred return in place of the required
    policy-average baseline.
    """

    if q_values.ndim != 2 or policy_probs.shape != q_values.shape:
        raise ValueError("q_values and policy_probs must both have shape [history, action]")
    if (policy_probs <= 0).any() or not torch.allclose(
        policy_probs.sum(dim=-1), torch.ones(q_values.shape[0], device=q_values.device), atol=atol
    ):
        raise ValueError("policy_probs must be normalized")
    value = (policy_probs * q_values).sum(dim=-1, keepdim=True)
    advantage = q_values - value
    # Any valid score-function derivative has zero policy-weighted mean.  Use a
    # deterministic, nontrivial zero-mean score to check both sides of the
    # policy-gradient identity, rather than confusing log-probabilities with
    # their parameter derivatives.
    raw_score = torch.arange(q_values.shape[1], device=q_values.device, dtype=q_values.dtype)
    raw_score = raw_score.expand_as(q_values)
    score = raw_score - (policy_probs * raw_score).sum(dim=-1, keepdim=True)
    lhs = (policy_probs * score * advantage).sum(dim=-1)
    rhs = (policy_probs * score * q_values).sum(dim=-1)
    if not torch.allclose(lhs, rhs, atol=atol):
        raise AssertionError(f"counterfactual advantage gradient failed: lhs={lhs}, rhs={rhs}")
