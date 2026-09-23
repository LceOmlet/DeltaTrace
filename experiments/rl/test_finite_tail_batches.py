"""Regression for the observed B1 paired-endpoint Dynamo constraint error.

This uses the real finite boundary formulas and Dynamo's eager backend to check
shape guards with exact eager arithmetic. It is not an FA/FLA tolerance test.
The real Qwen comparison separately exercises the installed Inductor backend.
"""
from unittest.mock import patch

import torch

from qwen35_decoder_finite import FiniteBoundaryOps
from qwen35_gdn_finite import _norm_gate_finite_rule, _conv_silu_finite_rule


def test_paired_tail_batches_do_not_force_a_symbolic_singleton():
    original_compile = torch.compile

    def compile_eager(fn, **kwargs):
        kwargs.pop('options', None)
        return original_compile(fn, backend='eager', **kwargs)

    with patch.object(torch, 'compile', compile_eager):
        ops = FiniteBoundaryOps(compiled=True, dynamic_shapes=True)
    torch.manual_seed(2026)
    for batch in (4, 2, 1, 4, 1):
        o, z = [torch.randn(2*batch, 64, 2, 16) for _ in range(2)]
        m, weight = torch.randn(batch, 64, 2, 16), torch.randn(16)
        args = (o, z, m, weight, 1e-6, 'content1')
        actual = ops.gdn_norm_gate(*args)
        torch.testing.assert_close(actual, _norm_gate_finite_rule(*args), rtol=0, atol=0)
        pre, output = [torch.randn(2*batch, 32, 64) for _ in range(2)]
        upstream = torch.randn(batch, 32, 64)
        args = (pre, output, upstream)
        torch.testing.assert_close(ops.gdn_conv_silu(*args), _conv_silu_finite_rule(*args), rtol=0, atol=0)
