"""Reproduce real PEFT CPU/meta dtype divergence and verify native defaults."""
import pytest
import torch
from types import SimpleNamespace
from peft import LoraConfig, get_peft_model
from verl.utils.fsdp_utils import get_init_weight_context_manager


def adapter_dtypes(rank, **kwargs):
    # Use VERL's actual CPU / Accelerate init_empty_weights contexts. Merely
    # constructing a Linear on device='meta' does not reproduce PEFT creation
    # while Accelerate intercepts parameter registration.
    mesh = SimpleNamespace(get_coordinate=lambda: [rank])
    context = get_init_weight_context_manager(use_meta_tensor=True, mesh=mesh)
    with context():
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16))
        model.to(torch.bfloat16)
        peft = get_peft_model(model, LoraConfig(r=1, lora_alpha=2, target_modules=['0']), **kwargs)
    return {p.dtype for n, p in peft.named_parameters() if 'lora_' in n}


@pytest.mark.parametrize('rank', [0, 1])
def test_native_adapter_casting_matches_across_initialization_devices(rank):
    assert adapter_dtypes(rank) == {torch.float32}


def test_previous_override_reproduces_the_dtype_mismatch():
    dtypes = [adapter_dtypes(rank, autocast_adapter_dtype=False) for rank in (0, 1)]
    assert dtypes == [{torch.bfloat16}, {torch.float32}]
