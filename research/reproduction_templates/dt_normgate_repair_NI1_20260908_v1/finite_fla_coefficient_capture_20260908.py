"""Expose the actual finite FLA write coefficients without another backward.

The caller verifies native source identities and restricts this backend to the
intended diagnostic layer. Each call returns (six_coefficients, packed_diagnostic)
where the latter contains only the existing L and r0 tensors, with layout
[B * H * ceil(T / 64), 64, V], packed in B/H/chunk order. They are neither
recomputed nor copied here. r0 is the production chunk-algebra readout from
native EOS endpoints, not a separately returned native intermediate.

The two unchanged native adjoint stages remain outside torch.compile, exactly
as in make_compiled_finite_pullback. Only the existing mixed contractions are
compiled. Extra output retention, compilation and any caller-side CPU capture
are diagnostic costs; no unchanged wall-time or bitwise scheduling claim is made.
"""
import torch

from finite_fla_gpu import mixed_coefficients, native_input_adjoints


def make_capture_backend():
    """Return the current direct-product backend plus its actual packed L/r0."""
    def mixed(endpoints, adjoints, scale):
        return mixed_coefficients(endpoints, adjoints, scale, False, diagnostics=True)

    compiled = torch.compile(mixed, fullgraph=True, dynamic=False,
        options={'triton.cudagraphs':False, 'max_autotune':False})

    def pullback(endpoints, do, scale):
        adjoints = native_input_adjoints(endpoints, do, scale)
        return compiled(endpoints, adjoints, scale)

    return pullback
