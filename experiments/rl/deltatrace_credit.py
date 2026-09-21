"""Inspect the official DT text-attribution interface without relabelling it.

The signed source vector is not automatically a same-prefix, policy-marginal
reward-event log ratio. Reward-event composition lives in counterfactual.py;
the old observation-routing and whole-return normalization are removed.
"""

from __future__ import annotations

from typing import Any

import torch

def _finite_scalar(value: Any, name: str) -> float:
    value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} is non-finite")
    return value


def trace_token_attribution(
    dt_runner: Any,
    reference_input_ids: torch.Tensor,
    selected_input_ids: torch.Tensor,
    target_case: dict[str, Any],
    target_offsets: list[int],
    *,
    packed_answer_targets: Any,
    # The owner runner reports the endpoint scalar in FP32 while the signed
    # pullback is accumulated in BF16/FP32 mixed kernels.  The pinned A6000
    # owner receipt measures a <=1.2% residual; keep a fail-closed 2% audit
    # envelope rather than silently renormalizing or inserting a baseline.
    attribution_tolerance: float = 2e-2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return the owner DT signed source-token attribution.

    ``reference_input_ids`` and ``selected_input_ids`` are the owner runner's
    equal-shaped reference/original endpoints. The returned signed tensor is
    shaped ``[1, sequence_length]``. It remains an owner artifact; no local
    attribution is reconstructed here.
    """

    if reference_input_ids.shape != selected_input_ids.shape or reference_input_ids.ndim != 2:
        raise ValueError("DT endpoints must have equal shape [1, sequence_length]")
    if reference_input_ids.shape[0] != 1:
        raise ValueError("one trajectory endpoint pair is required per trace call")
    pair = torch.cat((reference_input_ids, selected_input_ids), dim=0)
    selection = packed_answer_targets(
        [target_case], [target_offsets], pair.shape[1], selected_input_ids.device
    )
    try:
        signed, detail = dt_runner.attribute(
            pair, torch.ones_like(pair), selection, select_output_rows=True, observer=None
        )
    finally:
        release = getattr(getattr(dt_runner, "model", None), "release_owner_params", None)
        if callable(release):
            release()
    if signed.shape != reference_input_ids.shape:
        raise ValueError(
            f"owner DT returned {tuple(signed.shape)}, expected {tuple(reference_input_ids.shape)}"
        )
    if not torch.isfinite(signed).all():
        raise ValueError("owner DT signed attribution is non-finite")
    root_effect = _finite_scalar(detail["root_effect"], "DeltaTrace root_effect")
    signed_sum = float(signed.sum().item())
    if abs(root_effect - signed_sum) > attribution_tolerance * max(1.0, abs(root_effect)):
        raise AssertionError(
            "owner DT conservation failed: "
            f"root_effect={root_effect}, signed_sum={signed_sum}"
        )
    seed_effect = detail.get("compiled_seed_logprob_effect")
    if seed_effect is not None and abs(root_effect - float(seed_effect)) > attribution_tolerance * max(1.0, abs(root_effect)):
        raise AssertionError(
            "DeltaTrace scalar endpoint/seed mismatch: "
            f"root_effect={root_effect}, seed_effect={seed_effect}"
        )
    detail = dict(detail)
    detail.update(
        {
            "policy_credit_source": "text attribution diagnostic only; no reward-event log ratios",
            "policy_credit_signed_vector_used": False,
            "policy_credit_root_effect": root_effect,
            "policy_credit_signed_sum": signed_sum,
        }
    )
    return signed, torch.tensor([root_effect], device=selected_input_ids.device), detail
