"""Isolated row-contraction diagnostic ABI; never replaces production finite FA.

The original six operands, LSE, Q/K/V coefficients and three vendor-framework
phases are preserved. Four extra actual Q/K operands and two FP32 row outputs
are the entire ABI extension. No native attention forward/backward is called.
"""
import ctypes
import hashlib
from pathlib import Path

import torch

from vendor_fa_finite_bf16_d256 import RightPaddedLengths


class VendorFAConditionalDiagnostic:
    def __init__(self, library, expected_sha256):
        path = Path(library)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Diagnostic library hash mismatch.')
        self.library = ctypes.CDLL(str(path))
        self.operation = self.library.deltatrace_fa_finite_p1_bf16_d256_conditional_diag
        self.operation.argtypes = [ctypes.c_void_p] * 20 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        self.operation.restype = ctypes.c_int
        self.calls_entered = self.calls_returned = 0

    def __call__(self, operands, actual, scale, layout):
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
        tau = torch.empty((batch, heads, length), device=ref.device, dtype=torch.float32)
        center = torch.empty_like(tau)
        dq, dk, dv = (torch.empty_like(values[0]) for _ in range(3))
        actual_values = []
        for name in ('qc', 'kc', 'qa', 'ka'):
            value = actual[name]
            shape = (batch, kv_heads, length, dim) if name.startswith('k') else ref.shape
            assert value.shape == shape and value.device == ref.device and value.dtype == torch.bfloat16
            actual_values.append(value.detach().contiguous())
        fp32, bf16 = torch.empty_like(tau), torch.empty_like(tau)
        buffers = values + [tau, center, dq, dk, dv, layout._tensor] + actual_values + [fp32, bf16]
        assert len(buffers) == 20 and all(value.is_contiguous() for value in buffers)
        with torch.cuda.device(ref.device):
            self.calls_entered += 1
            status = self.operation(*[ctypes.c_void_p(value.data_ptr()) for value in buffers],
                batch, heads, kv_heads, length, scale,
                ctypes.c_void_p(torch.cuda.current_stream(ref.device).cuda_stream))
            torch.cuda.synchronize()
        if status != 0:
            raise RuntimeError(f'Diagnostic finite FA launch failed: {status}')
        self.calls_returned += 1
        result = {'dq': dq, 'dk': dk, 'dv': dv, 'tau': tau, 'center': center,
                  'conditional_fp32': fp32, 'conditional_bf16': bf16}
        assert all(bool(torch.isfinite(value).all()) for value in result.values())
        return result
