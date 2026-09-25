"""Diagnostic-only dtype boundary around the original FLA public operator.

No recurrence, cache update, normalization, or backward is implemented here.
Qwen's existing operator attribute supplies the native call. Model weights and
the Qwen input/output dtype stay unchanged. The context restores every binding.
"""
from contextlib import contextmanager
from functools import wraps
import torch


@contextmanager
def native_fla_fp16(model):
    restored = []
    try:
        for module in model.modules():
            if not hasattr(module, 'chunk_gated_delta_rule'):
                continue
            original = module.chunk_gated_delta_rule

            def make_boundary(owner):
                @wraps(owner)
                def boundary(q, k, v, g, beta, **kwargs):
                    o, state = owner(q.to(torch.float16), k.to(torch.float16),
                                     v.to(torch.float16), g=g,
                                     beta=beta.to(torch.float16), **kwargs)
                    return o.to(v.dtype), state
                return boundary

            restored.append((module, original))
            module.chunk_gated_delta_rule = make_boundary(original)
        yield
    finally:
        for module, original in reversed(restored):
            module.chunk_gated_delta_rule = original
