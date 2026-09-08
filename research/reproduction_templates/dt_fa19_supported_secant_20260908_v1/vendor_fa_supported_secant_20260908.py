"""One explicitly named finite-rule candidate, using the pinned FA framework.

The model/native FA, production finite library, QK midpoint and raw P1 value
branch are untouched. Route probabilities are explicitly normalized per row.
No alternate attention forward, fallback, clipping or denominator epsilon.
"""
import ctypes
import hashlib
from pathlib import Path

import torch

from vendor_fa_finite_bf16_d256 import RightPaddedLengths


ROW_FIELDS = ('s_anchor', 'g_anchor', 'g_mean_offset', 'inverse_p0_sum', 'inverse_p1_sum', 'kappa', 'raw_Jeffreys_denominator', 'numerator', 'raw_p0_sum', 'raw_p1_sum', 'raw_route_target', 'normalized_route_target', 'endpoint_positive', 'endpoint_negative', 'base_gradient_contraction', 'normalized_direction_contraction', 'route_row_sum', 'osc_g', 'direction_row_sum', 'correction_positive', 'correction_negative')


class VendorFASupportedSecant:
    def __init__(self, library, expected_sha256):
        path = Path(library)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Supported-secant library hash mismatch.')
        self.library = ctypes.CDLL(str(path))
        self.operation = self.library.deltatrace_fa_finite_p1_supported_secant
        self.operation.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        self.operation.restype = ctypes.c_int
        self.calls_entered = self.calls_returned = 0

    def __call__(self, operands, scale, layout):
        ref = operands['q0']
        batch, heads, length, dim = ref.shape
        assert dim == 256 and ref.is_cuda and ref.dtype == torch.bfloat16
        assert isinstance(layout, RightPaddedLengths)
        assert len(layout.lengths) == batch and layout.padded_length == length
        assert layout._tensor.device == ref.device
        kv_heads = operands['k0'].shape[1]
        assert kv_heads > 0 and heads % kv_heads == 0
        values = []
        for name in ('q0', 'k0', 'q1', 'k1', 'v0', 'u'):
            value = operands[name]
            shape = (batch, kv_heads, length, dim) if name in ('k0', 'k1', 'v0') else ref.shape
            assert value.shape == shape and value.device == ref.device
            assert value.dtype == (torch.float32 if name == 'u' else torch.bfloat16)
            values.append(value.detach().to(dtype=torch.bfloat16).contiguous())
        for name in ('lse0', 'lse1'):
            value = operands[name]
            assert value.shape == (batch, heads, length) and value.device == ref.device
            assert value.dtype == torch.float32
            values.append(value.detach().contiguous())
        rows = torch.empty((len(ROW_FIELDS), batch, heads, length), device=ref.device, dtype=torch.float32)
        dq, dk, dv = (torch.empty_like(values[0]) for _ in range(3))
        buffers = values + [rows, dq, dk, dv, layout._tensor]
        assert len(buffers) == 13 and all(value.is_contiguous() for value in buffers)
        with torch.cuda.device(ref.device):
            self.calls_entered += 1
            status = self.operation(*[ctypes.c_void_p(value.data_ptr()) for value in buffers],
                batch, heads, kv_heads, length, scale,
                ctypes.c_void_p(torch.cuda.current_stream(ref.device).cuda_stream))
            torch.cuda.synchronize()
        if status != 0:
            raise RuntimeError(f'Supported-secant finite FA launch failed: {status}')
        self.calls_returned += 1
        return {'dq': dq, 'dk': dk, 'dv': dv, 'rows': rows}
