"""A guarded DeltaTrace-to-policy-credit adapter.

DeltaTrace's signed input scores are not policy advantages.  This adapter
uses the existing runner only to obtain and audit the scalar endpoint return
difference for two action interventions.  The scalar difference is the
counterfactual ``G(h,a)-G(h,a')``; the per-input signed vector is retained for
diagnostics and is never used as a token-wise policy-loss multiplier.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from .counterfactual import CounterfactualCredit, averaged_counterfactual_credit
except ImportError:  # direct execution from the remote experiment directory
    from counterfactual import CounterfactualCredit, averaged_counterfactual_credit


def _finite_scalar(value: Any, name: str) -> float:
    value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} is non-finite")
    return value


def trace_endpoint_difference(
    dt_runner: Any,
    reference_input_ids: torch.Tensor,
    selected_input_ids: torch.Tensor,
    target_case: dict[str, Any],
    target_offsets: list[int],
    *,
    packed_answer_targets: Any,
    attribution_tolerance: float = 5e-3,
) -> tuple[float, dict[str, Any]]:
    """Run the owner DeltaTrace runner for one counterfactual pair.

    ``reference_input_ids`` and ``selected_input_ids`` must have the same
    shape and differ at the action intervention.  ``packed_answer_targets`` is
    the checked ``PackedAnswerTargets`` class imported from the owner source
    tree.  The runner's scalar ``root_effect`` is returned only after it agrees
    with the runner's compiled scalar seed.  The signed input vector's
    residual is recorded as an attribution diagnostic; it is not silently
    treated as an advantage and is never required to be zero for the policy
    signal.
    """

    if reference_input_ids.shape != selected_input_ids.shape or reference_input_ids.ndim != 2:
        raise ValueError("counterfactual endpoints must be equal-shaped [1, length] tensors")
    if reference_input_ids.shape[0] != 1:
        raise ValueError("one selected/reference action pair is required per trace call")
    pair = torch.cat((reference_input_ids, selected_input_ids), dim=0)
    selection = packed_answer_targets(
        [target_case], [target_offsets], pair.shape[1], selected_input_ids.device
    )
    signed, detail = dt_runner.attribute(
        pair, torch.ones_like(pair), selection, select_output_rows=True, observer=None
    )
    signed_sum = float(signed.sum().item())
    root_effect = _finite_scalar(detail["root_effect"], "DeltaTrace root_effect")
    seed_effect = detail.get("compiled_seed_logprob_effect")
    if seed_effect is not None and abs(root_effect - float(seed_effect)) > attribution_tolerance * max(1.0, abs(root_effect)):
        raise AssertionError(
            "DeltaTrace scalar endpoint/seed mismatch: "
            f"root_effect={root_effect}, seed_effect={seed_effect}"
        )
    detail = dict(detail)
    detail["policy_credit_source"] = "DeltaTrace scalar endpoint difference"
    detail["policy_credit_signed_vector_used"] = False
    detail["policy_credit_root_effect"] = root_effect
    detail["policy_credit_signed_residual"] = root_effect - signed_sum
    detail["policy_credit_signed_residual_relative"] = (
        (root_effect - signed_sum) / root_effect if root_effect else None
    )
    return root_effect, detail


def averaged_traced_credit(
    selected_effect: torch.Tensor, reference_effects: torch.Tensor
) -> CounterfactualCredit:
    """Build the policy credit after independently tracing reference actions."""

    return averaged_counterfactual_credit(selected_effect, reference_effects)
