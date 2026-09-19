"""Minimal LoRA wrappers for the existing Transformers model.

The remote environment intentionally has no PEFT/TRL installation.  This is a
small training-only adapter; it leaves the base checkpoint frozen and does not
modify the installed Transformers or DeltaTrace owner code.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the effective linear weight to DeltaTrace's finite rules.

        With dropout disabled, the LoRA branch is exactly linear, so
        ``W_eff = W_base + scale * B @ A`` is the correct transpose map for
        the owner attribution pullback.  This property keeps the wrapped
        module compatible with the existing capture code without changing it.
        """

        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + update * self.scaling


def _replace(parent: nn.Module, child_name: str, wrapped: nn.Module) -> None:
    if child_name.isdigit():
        parent[int(child_name)] = wrapped  # type: ignore[index]
    else:
        setattr(parent, child_name, wrapped)


def inject_lora(
    model: nn.Module,
    target_suffixes: Iterable[str] = ("q_proj", "v_proj"),
    *,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> list[nn.Parameter]:
    """Wrap selected ``nn.Linear`` modules and return trainable parameters."""

    suffixes = tuple(target_suffixes)
    if not suffixes:
        raise ValueError("at least one target suffix is required")
    model.requires_grad_(False)
    candidates: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if ".base" in name:
            continue
        if "." not in name:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        candidates.append((name, parent, child_name, module))
    for _, parent, child_name, module in candidates:
        _replace(parent, child_name, LoRALinear(module, rank, alpha, dropout))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"no Linear modules matched target suffixes {suffixes}")
    return params


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
