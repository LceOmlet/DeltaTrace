"""BF16/D256 content_P1 finite propagation on the pinned vendor FA framework.

This is a separately named attribution operator. It does not replace the model
forward/backward or read private attention buffers. K/V inputs remain compact;
K/V outputs are query-head coefficients, reduced in FP32 by the caller.
"""
import ctypes
import hashlib
from pathlib import Path

import torch


class RightPaddedLengths:
    """Validate CPU lengths once; reuse their device buffer without per-call sync.

    Only contiguous right padding, equal Q/K lengths, no packed/cache sequences.
    The private tensor must not be mutated while this layout is in use.
    """
    def __init__(self, lengths, padded_length, device):
        self.lengths = tuple(lengths)
        if not self.lengths or not all(type(n) is int and 0 < n <= padded_length for n in self.lengths):
            raise ValueError('Require nonempty positive integer lengths within the padded extent.')
        self.padded_length = padded_length
        self._tensor = torch.tensor(self.lengths, dtype=torch.int32, device=device)
        if not self._tensor.is_cuda:
            raise ValueError('The native finite operator requires a CUDA/MACA device.')


class VendorFAFiniteP1BF16D256:
    def __init__(self, library, expected_sha256):
        path = Path(library)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Finite FA library hash mismatch.')
        self.library = ctypes.CDLL(str(path))
        self.operation = self.library.deltatrace_fa_finite_p1_bf16_d256
        self.operation.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        self.operation.restype = ctypes.c_int

    def __call__(self, operands, scale, layout, activity=None):
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
            expected = (batch, kv_heads, length, dim) if name in ('k0', 'k1', 'v0') else ref.shape
            assert value.shape == expected and value.device == ref.device
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
        buffers = values + [tau, center, dq, dk, dv, layout._tensor]
        if activity is not None:
            activity.update(query_heads=heads, kv_heads=kv_heads, valid_lengths=list(layout.lengths),
                padded_length=length, GQA_input_expansion=False, global_endpoint_mean_buffers=0,
                global_attention_matrix=False, kernel_launches_per_call=3,
                endpoint_mean='FP32 add then BF16 storage in existing FA shared tile',
                buffer_contract=[{'shape': list(v.shape), 'dtype': str(v.dtype),
                                  'bytes': v.numel() * v.element_size()} for v in buffers])
            activity['calls_attempted'] = activity.get('calls_attempted', 0) + 1
        with torch.cuda.device(ref.device):
            status = self.operation(*[ctypes.c_void_p(v.data_ptr()) for v in buffers],
                batch, heads, kv_heads, length, scale,
                ctypes.c_void_p(torch.cuda.current_stream(ref.device).cuda_stream))
        if status != 0:
            raise RuntimeError(f'Finite FA launch failed: {status}')
        if activity is not None:
            activity['calls_enqueued'] = activity.get('calls_enqueued', 0) + 1
        return {'dq': dq, 'dk': dk, 'dv': dv, 'tau': tau, 'center': center}
