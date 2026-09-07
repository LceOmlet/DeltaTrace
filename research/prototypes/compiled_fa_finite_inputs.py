"""Fuse existing finite-FA input preparation with public torch.compile.

This is a layout/cast/midpoint helper, not attention, a GEMM, or model backward.
Expressions match the compact-GQA runtime; no reduced-precision policy change.
The actual FA finite shared library and its three passes are not replaced.
"""
import torch


def prepare_rule(q0, k0, q1, k1, v0, u):
    values = [x.detach().to(dtype=torch.float16).contiguous()
              for x in (q0, k0, q1, k1, v0, u)]
    values.append(((q0.float() + q1.float()) * .5).half().contiguous())
    values.append(((k0.float() + k1.float()) * .5).half().contiguous())
    return tuple(values)


_prepare = torch.compile(prepare_rule, fullgraph=True, dynamic=True, backend='inductor')


def prepare_finite_fa_inputs(*args):
    with torch.profiler.record_function('ATTR_COMPILED_FINITE_FA_INPUTS'):
        return _prepare(*args)
