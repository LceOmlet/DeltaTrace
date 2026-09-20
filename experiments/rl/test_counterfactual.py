"""Small invariant tests for the counterfactual policy-credit contract."""

from __future__ import annotations

import torch

from counterfactual import assert_advantage_equivalence, averaged_counterfactual_credit, score_function_loss
from deltatrace_credit import trace_endpoint_difference


def main() -> None:
    torch.manual_seed(0)
    q = torch.tensor([[1.0, 3.0, -2.0], [4.0, 0.5, 2.0]])
    probs = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    assert_advantage_equivalence(q, probs)

    # Non-degenerate cases are required here: an equal-return fixture would
    # only test the zero-credit special case and could not catch a sign or
    # leave-one-out bug.
    selected = torch.tensor([4.0, -1.0])
    references = torch.tensor([[1.0, 3.0, 5.0], [4.0, 0.0, 2.0]])
    batch = averaged_counterfactual_credit(selected, references)
    expected = torch.tensor([1.0, -3.0])
    assert torch.allclose(batch.credit, expected)
    log_prob = torch.tensor([-0.4, -0.7], requires_grad=True)
    loss = score_function_loss(log_prob, batch.credit)
    loss.backward()
    assert torch.allclose(log_prob.grad, torch.tensor([-0.5, 1.5]))

    # The collector's leave-one-out contract must exclude the selected
    # rollout from its reference mean.  For returns [1, 0, -1], the credits
    # are [1.5, 0, -1.5], not a group-centred average with self included.
    loo_selected = torch.tensor([1.0, 0.0, -1.0])
    loo_references = torch.tensor([[0.0, -1.0], [1.0, -1.0], [1.0, 0.0]])
    loo = averaged_counterfactual_credit(loo_selected, loo_references)
    assert torch.allclose(loo.credit, torch.tensor([1.5, 0.0, -1.5]))
    class FakeSelection:
        def __init__(self, cases, offsets, length, device):
            assert cases and offsets == [[0]] and length == 3

    class FakeRunner:
        def attribute(self, pair, mask, selection, select_output_rows=True, observer=None):
            assert pair.shape == (2, 3) and bool(mask.eq(1).all())
            return torch.tensor([1.25]), {"root_effect": 1.25}

    effect, detail = trace_endpoint_difference(
        FakeRunner(), torch.tensor([[1, 2, 3]]), torch.tensor([[1, 9, 3]]),
        {"target_ids": torch.tensor([7]), "prompt_length": 1}, [0],
        packed_answer_targets=FakeSelection,
    )
    assert effect == 1.25 and detail["policy_credit_signed_vector_used"] is False
    print({
        "status": "passed",
        "credit": batch.credit.tolist(),
        "loo_credit": loo.credit.tolist(),
        "loss": float(loss.detach()),
        "grad": log_prob.grad.tolist(),
    })


if __name__ == "__main__":
    main()
