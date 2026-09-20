"""Exercise the patched upstream VERL counterfactual advantage boundary."""

from __future__ import annotations

import torch

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage


def main() -> None:
    # A nonzero first-row credit must reach its response tokens.  The second
    # row has a nonzero credit but is a later-action mask, so it must be zero.
    data = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((2, 3), dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
            "counterfactual_credit": torch.tensor([2.0, -1.0]),
            "counterfactual_action_mask": torch.tensor([True, False]),
        }
    )
    out = compute_advantage(data, AdvantageEstimator.COUNTERFACTUAL)
    expected = torch.tensor([[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    assert torch.equal(out.batch["advantages"], expected)
    assert torch.equal(out.batch["returns"], expected)
    print({"status": "passed", "advantages": out.batch["advantages"].tolist()})


if __name__ == "__main__":
    main()
