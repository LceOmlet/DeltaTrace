"""Test the dispatch seam against the actual pinned actor, not a copied loss."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from verl.workers.actor import dp_actor
from patch_actor_entropy_dispatch import patch


def test_only_dense_dispatch_changes():
    source = Path(dp_actor.__file__).read_text()
    old = source.replace('entropy = self.compute_entropy_from_logits(logits)',
                         'entropy = verl_F.entropy_from_logits(logits)')
    fixed = patch(old)
    expected = old.replace('entropy = verl_F.entropy_from_logits(logits)',
                           'entropy = self.compute_entropy_from_logits(logits)')
    assert fixed == expected
    assert patch(fixed) == fixed
    def methods(s):
        cls = next(n for n in ast.parse(s).body if isinstance(n, ast.ClassDef) and n.name == 'DataParallelPPOActor')
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    before, after = methods(old), methods(fixed)
    assert {k for k in before if before[k] != after[k]} == {'_forward_micro_batch'}


def test_dense_path_calls_configured_owner_entropy(monkeypatch):
    from test_shared_padding import HeadOnlyModel
    import verl.utils.torch_functional as functional
    monkeypatch.setattr(functional, 'FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE', False)
    monkeypatch.setenv('VERL_TRIM_SHARED_PADDING', '0')
    monkeypatch.setenv('VERL_TRIM_RESPONSE_HEAD', '0')
    calls = []
    def record(logits):
        calls.append(logits.shape)
        return functional.entropy_from_logits(logits)
    actor = SimpleNamespace(actor_module=HeadOnlyModel(), device_name='cpu',
        use_remove_padding=False, use_fused_kernels=False, compute_entropy_from_logits=record)
    ids = torch.arange(16).reshape(2, 8) % 8
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 position_ids=torch.arange(8).expand(2, -1), responses=ids[:, -4:])
    dp_actor.DataParallelPPOActor._forward_micro_batch(actor, batch, 1., False)
    assert not calls
    entropy, lp = dp_actor.DataParallelPPOActor._forward_micro_batch(actor, batch, 1., True)
    assert calls == [torch.Size([2, 4, 8])]
    assert entropy.shape == lp.shape == (2, 4)
