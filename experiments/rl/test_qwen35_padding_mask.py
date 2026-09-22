"""Backported mask behavior versus archived official Transformers sources."""
import ast
import os
from pathlib import Path

import torch

from patch_verl_agent2 import patch_qwen35_padding_mask


def owner_function(source):
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'apply_mask_to_padding_states')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<official mask source>', 'exec'), namespace)
    return node, namespace[node.name]


def test_backport_matches_current_official_function():
    previous = Path(os.environ['QWEN35_PADDING_PREVIOUS_SOURCE']).read_text()
    official = Path(os.environ['QWEN35_PADDING_REFERENCE_SOURCE']).read_text()
    patched = patch_qwen35_padding_mask(previous)
    assert ast.dump(owner_function(patched)[0]) == ast.dump(owner_function(official)[0])
    assert patch_qwen35_padding_mask(patched) == patched


def test_batch_one_matches_existing_owner_batch_two_forward_and_gradient():
    previous = Path(os.environ['QWEN35_PADDING_PREVIOUS_SOURCE']).read_text()
    old = owner_function(previous)[1]
    fixed = owner_function(patch_qwen35_padding_mask(previous))[1]
    x = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).requires_grad_()
    mask = torch.tensor([[0, 0, 1, 1, 1, 0]])
    duplicated = x.detach().repeat(2, 1, 1).requires_grad_()
    expected = old(duplicated, mask.repeat(2, 1))[:1]
    actual = fixed(x, mask)
    assert not torch.equal(old(x, mask), expected)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), duplicated)[0][:1], atol=0, rtol=0)
    torch.testing.assert_close(fixed(duplicated, mask.repeat(2, 1)), old(duplicated, mask.repeat(2, 1)), atol=0, rtol=0)
    assert fixed(x, None) is x
