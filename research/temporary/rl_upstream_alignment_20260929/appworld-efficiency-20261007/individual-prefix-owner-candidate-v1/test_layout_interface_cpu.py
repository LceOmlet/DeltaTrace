"""CPU-only layout/ABI checks. No Torch runtime, model or numerical acceptance.

The tensor stand-ins exercise the owner's Python argument/shape transport only.
Integer address checks exercise the changed CUDA indexing, not its arithmetic.
"""
import contextlib
import ctypes
import re
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY_NAME = 'vendor_fa_finite_bf16_d256.py'
CU_NAME = 'vendor_fa_finite_p1_bf16_d256.cu'


class Tensor:
    next_pointer = 4096

    def __init__(self, shape, dtype, device='cuda'):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.is_cuda = device == 'cuda'
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += 4096

    def detach(self):
        return self

    def to(self, *, dtype):
        return self if dtype == self.dtype else Tensor(self.shape, dtype, self.device)

    def contiguous(self):
        return self

    def data_ptr(self):
        return self.pointer

    def numel(self):
        value = 1
        for n in self.shape:
            value *= n
        return value

    def element_size(self):
        return 2 if self.dtype == 'bf16' else 4


TORCH = types.SimpleNamespace(
    int32='i32', float32='f32', bfloat16='bf16',
    tensor=lambda values, dtype, device: Tensor((len(values),), dtype, device),
    empty=lambda shape, device, dtype: Tensor(shape, dtype, device),
    empty_like=lambda value: Tensor(value.shape, value.dtype, value.device),
    cuda=types.SimpleNamespace(
        device=lambda device: contextlib.nullcontext(),
        current_stream=lambda device: types.SimpleNamespace(cuda_stream=17)),
)


def load_owner(subdirectory):
    source = (ROOT / subdirectory / PY_NAME).read_text(encoding='utf-8')
    # This CPU test intentionally does not import the local Torch/CUDA runtime.
    source = source.replace('import torch\n', '')
    namespace = {'torch': TORCH}
    exec(compile(source, str(ROOT / subdirectory / PY_NAME), 'exec'), namespace)
    return namespace


class Operation:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return 0


class LayoutInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_owner('baseline')
        cls.new = load_owner('candidate')

    def invoke(self, owner, lengths, extent, *, query_stride=None, **kwargs):
        layout = owner['RightPaddedLengths'](lengths, extent, 'cuda', **kwargs)
        batch, heads, kv_heads, dim = len(lengths), 4, 2, 256
        qlen = query_stride if query_stride is not None else extent - kwargs.get('query_start', 0)
        operands = {name: Tensor((batch, heads, qlen, dim), 'f32' if name == 'u' else 'bf16')
                    for name in ('q0', 'q1', 'u')}
        operands.update({name: Tensor((batch, kv_heads, extent, dim), 'bf16')
                         for name in ('k0', 'k1', 'v0')})
        operands.update({name: Tensor((batch, heads, qlen), 'f32') for name in ('lse0', 'lse1')})
        names = ('', '_suffix', '_cached_suffix', '_row_cached_suffix')
        library = types.SimpleNamespace(**{
            'deltatrace_fa_finite_p1_bf16_d256' + name: Operation() for name in names})
        operation = owner['VendorFAFiniteP1BF16D256'].__new__(owner['VendorFAFiniteP1BF16D256'])
        operation.library = library
        operation.operation = library.deltatrace_fa_finite_p1_bf16_d256
        outputs = operation(operands, 0.0625, layout)
        called = [(name, getattr(library, 'deltatrace_fa_finite_p1_bf16_d256' + name))
                  for name in names if getattr(library, 'deltatrace_fa_finite_p1_bf16_d256' + name).calls]
        self.assertEqual(len(called), 1)
        return layout, outputs, called[0]

    def test_original_default_suffix_and_cached_export_transport_unchanged(self):
        for kwargs in ({}, {'coefficient_starts': [0, 64, 128, 200]},
                       {'query_start': 64, 'coefficient_starts': [64, 64, 128, 200]}):
            old = self.invoke(self.old, [128, 192, 224, 240], 256, **kwargs)
            new = self.invoke(self.new, [128, 192, 224, 240], 256, **kwargs)
            self.assertEqual(old[2][0], new[2][0])
            scalar_args = lambda call: tuple(x.value if isinstance(x, ctypes.c_void_p) else x for x in call[-6:])
            self.assertEqual(scalar_args(old[2][1].calls[0]), scalar_args(new[2][1].calls[0]))
            self.assertEqual({k: v.shape for k, v in old[1].items()},
                             {k: v.shape for k, v in new[1].items()})
            self.assertIsNone(new[0].query_starts)

    def test_b4_row_export_separates_query_and_kv_strides(self):
        layout, outputs, (name, operation) = self.invoke(
            self.new, [100, 90, 200, 224], 224, query_stride=128,
            query_starts=[0, 64, 128, 192], query_padded_length=128,
            coefficient_starts=[0, 64, 128, 192])
        self.assertEqual(name, '_row_cached_suffix')
        self.assertEqual(operation.argtypes,
                         [ctypes.c_void_p] * 16 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p])
        self.assertEqual(operation.calls[0][16:21], (4, 4, 2, 224, 128))
        self.assertEqual(operation.calls[0][15].value, layout._query_starts.data_ptr())
        self.assertEqual(outputs['dk'].shape, (4, 4, 128, 256))

    def test_invalid_row_layouts_are_rejected_before_launch(self):
        base = {'lengths': [100, 90, 200, 224], 'padded_length': 224, 'device': 'cuda',
                'query_starts': [0, 64, 128, 192], 'query_padded_length': 128,
                'coefficient_starts': [0, 64, 128, 192]}
        changes = ({'query_start': 64}, {'query_starts': [0, 64]},
                   {'query_starts': [0, 63, 128, 192]}, {'query_starts': [0, 128, 128, 192]},
                   {'query_padded_length': 99}, {'query_padded_length': 0},
                   {'query_padded_length': None}, {'coefficient_starts': None},
                   {'coefficient_starts': [0, 0, 128, 192]})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.new['RightPaddedLengths'](**(base | change))
        with self.assertRaises(ValueError):
            self.new['RightPaddedLengths']([100], 128, 'cuda', query_padded_length=100)

    def test_integer_index_transport_and_every_right_pad_slot(self):
        for starts, qstride, kvstride in (([0, 64, 128, 192], 128, 224),
                                         ([64, 128, 192, 256], 65, 300)):
            heads, kvheads, dim = 4, 2, 256
            tiles = (qstride + 63) // 64
            for row, start in enumerate(starts):
                local_written = [tile * 64 + j for tile in range(tiles) for j in range(64)
                                 if start + tile * 64 + j < start + qstride]
                self.assertEqual(local_written, list(range(qstride)))
                for head in range(heads):
                    bh = row * heads + head
                    offset = (bh * qstride - start) * dim
                    for local in range(qstride):
                        for d in (0, dim - 1):
                            self.assertEqual(offset + (start + local) * dim + d,
                                             (bh * qstride + local) * dim + d)
                    kv_bh = row * kvheads + head // (heads // kvheads)
                    self.assertEqual(kv_bh * kvstride * dim,
                                     (row * kvheads + head // 2) * kvstride * dim)
            self.assertTrue(any(start + qstride > kvstride for start in starts))

    def test_original_c_exports_remain_exact_and_single_kernel_is_reused(self):
        old = (ROOT / 'baseline' / CU_NAME).read_text(encoding='utf-8')
        new = (ROOT / 'candidate' / CU_NAME).read_text(encoding='utf-8')
        exports = re.findall(r'extern "C" int deltatrace_fa_finite_p1_bf16_d256(?:_suffix|_cached_suffix)?\([\s\S]*?return launch_finite\(p,stream_ptr\);\n}', old)
        self.assertEqual(len(exports), 3)
        for exported in exports:
            self.assertIn(exported, new)
        self.assertEqual(new.count('__global__ void deltatrace_fa_finite_p1_kernel'), 1)
        self.assertEqual(new.count('deltatrace_fa_finite_p1_kernel<0><<<'), 1)
        self.assertEqual(new.count('deltatrace_fa_finite_p1_kernel<1,CachedTraits><<<'), 1)
        self.assertEqual(new.count('deltatrace_fa_finite_p1_kernel<2,CachedTraits><<<'), 1)
        arithmetic = old[old.index('                const float lp0='):old.index('            } else if constexpr(Phase!=0)')]
        self.assertIn(arithmetic, new)


if __name__ == '__main__':
    unittest.main(verbosity=2)
