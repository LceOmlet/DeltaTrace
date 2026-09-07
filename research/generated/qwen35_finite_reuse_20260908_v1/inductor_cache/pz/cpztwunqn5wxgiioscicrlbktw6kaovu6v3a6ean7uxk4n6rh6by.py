# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/kq/ckqdwz4zggab55od37ozccpmvfrh7yukt7hp676x73xn2k3vuodd.py
# Topologically Sorted Source Nodes: [a_3], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   a_3 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_1, [0, 0, 0, 63], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_poi_fused_constant_pad_nd_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3145728, 'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 192)
    x3 = xindex
    y1 = ((yindex // 192) % 32)
    y2 = yindex // 6144
    y4 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (y0 + 129*x3 + 16512*y1 + 1056768*y2), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x3 + 128*y4), tmp3, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/r7/cr7ydmqupofa4n2fvv4yyi4cve4web5lljxalflsotujpsfb432x.py
# Topologically Sorted Source Nodes: [reshape_12, reshape_13, float_8, float_9, mul_20, S], Original ATen: [aten.clone, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   S => sum_3
#   float_8 => convert_element_type_41
#   float_9 => convert_element_type_42
#   mul_20 => mul_20
#   reshape_12 => clone
#   reshape_13 => clone_1
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_13,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.float32), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_41, %convert_element_type_42), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [-1, -2]), kwargs = {})
triton_red_fused__to_copy_clone_mul_sum_1 = async_compile.triton('triton_red_fused__to_copy_clone_mul_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072, 'r0_': 37748736}}
)
@triton.jit
def triton_red_fused__to_copy_clone_mul_sum_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 384
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 3)
    x2 = ((xindex // 6) % 32)
    x3 = xindex // 192
    x5 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_4 + 8192*x0 + 16384*x2 + 524288*x1 + 3145728*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_4 + 8192*x0 + 16384*x2 + 524288*x1 + 1572864*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0.to(tl.float32)
        tmp3 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tl.store(out_ptr0 + (r0_4 + 8192*x5), tmp0, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_4 + 8192*x5), tmp1, r0_mask & xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/w4/cw4kxmkxktdc7wrngv63c6o3lmr5esfaibd3z6d54jktv2cgqpjp.py
# Topologically Sorted Source Nodes: [a_21], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   a_21 => constant_pad_nd_8
# Graph fragment:
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_10, [0, 0, 0, 63], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_2 = async_compile.triton('triton_poi_fused_constant_pad_nd_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 128) % 192)
    x0 = (xindex % 128)
    x2 = ((xindex // 24576) % 32)
    x3 = xindex // 786432
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 128*x2 + 4096*x1 + 528384*x3), tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp3, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/uq/cuqtachtc7b2o3h7kay4c62qpujugag6zc5n6i6u3sziunp5gzh6.py
# Topologically Sorted Source Nodes: [a_1, float_10, mul_21, sum_4], Original ATen: [aten.constant_pad_nd, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   a_1 => constant_pad_nd
#   float_10 => convert_element_type_43
#   mul_21 => mul_21
#   sum_4 => sum_4
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute, [0, 0, 0, 63], 0.0), kwargs = {})
#   %convert_element_type_43 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_43, %bmm_6), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_21, [-1]), kwargs = {})
triton_red_fused__to_copy_constant_pad_nd_mul_sum_3 = async_compile.triton('triton_red_fused__to_copy_constant_pad_nd_mul_sum_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_constant_pad_nd_mul_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3244032, 'r0_': 12582912}}
)
@triton.jit
def triton_red_fused__to_copy_constant_pad_nd_mul_sum_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 32)
    x2 = xindex // 6144
    x4 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp5 = tl.load(in_ptr1 + (r0_3 + 128*x4), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 129, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (528384 + x0 + 129*r0_3 + 16512*x1 + 1056768*x2), r0_mask & tmp2, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tl.store(out_ptr0 + (r0_3 + 128*x4), tmp3, r0_mask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/gk/cgkamyu2etowsnv7oelkujkbhmilbflsh63puh2z53aery4656sl.py
# Topologically Sorted Source Nodes: [a_5], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   a_5 => constant_pad_nd_2
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_2, [0, 0, 0, 63], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_4 = async_compile.triton('triton_poi_fused_constant_pad_nd_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3145728, 'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 192)
    x3 = xindex
    y1 = ((yindex // 192) % 32)
    y2 = yindex // 6144
    y4 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (528384 + y0 + 129*x3 + 16512*y1 + 1056768*y2), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x3 + 128*y4), tmp3, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/fs/cfsm5rvzi2o66bdyet56tcqasycwd2hxqoxfmrcd6266f5zjrfi2.py
# Topologically Sorted Source Nodes: [a_19], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   a_19 => constant_pad_nd_7
# Graph fragment:
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_9, [0, 0, 0, 63], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_5 = async_compile.triton('triton_poi_fused_constant_pad_nd_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1572864, 'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 192)
    x3 = xindex
    y1 = ((yindex // 192) % 32)
    y2 = yindex // 6144
    y4 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (264192 + y0 + 129*x3 + 8256*y1 + 528384*y2), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x3 + 64*y4), tmp3, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/wf/cwfsjwfusomkah6rbzy23uk3eudrt7hwi47biwa6srvy6zulg636.py
# Topologically Sorted Source Nodes: [W, to_5, to_14], Original ATen: [aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   W => mul
#   to_14 => convert_element_type_16
#   to_5 => convert_element_type_5
# Graph fragment:
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %bmm), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_17, torch.bfloat16), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_mul_6 = async_compile.triton('triton_poi_fused__to_copy_mul_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 128
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp0 = ((x2 // 128) % 192)
    tmp1 = tl.full([1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4128 + 32*((x1 % 192)) + 8256*(x1 // 6144) + (((x1 // 192) % 32))), tmp2, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr1 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/6b/c6bzeji6n6vdkny6ht23crbxg7my2maeyw7kt5635ffjk2x5s6ti.py
# Topologically Sorted Source Nodes: [triton_kernel_wrapper_mutation, sub_1, clamp_max_1, exp_1, E1, mul_7, to_20, mul_9, to_22], Original ATen: [aten.sub, aten.clamp_max, aten.exp, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   E1 => mul_2
#   clamp_max_1 => clamp_max_1
#   exp_1 => exp_1
#   mul_7 => mul_7
#   mul_9 => mul_9
#   sub_1 => sub_1
#   to_20 => convert_element_type_24
#   to_22 => convert_element_type_27
#   triton_kernel_wrapper_mutation => triton_kernel_wrapper_mutation
# Graph fragment:
#   %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 0, grid: [(192, 1, 1)], tma_descriptor_metadata: {}, kwargs: {M: %sub_8, RawG0: %view_5, G0: %view_6, G1: %view_7, Bterm: %sub_7, Dterm: %sum_6, Sterm: %sum_3, Out: %permute_26}})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_7, %unsqueeze_8), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%sub_1, 0), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%clamp_max_1,), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %permute_15), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_1, %mul_2), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_2, %mul_2), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_clamp_max_exp_mul_sub_7 = async_compile.triton('triton_poi_fused__to_copy_clamp_max_exp_mul_sub_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_max_exp_mul_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_max_exp_mul_sub_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 64)
    x3 = xindex // 4096
    x2 = ((xindex // 64) % 64)
    x4 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 - tmp7
    tmp9 = ((x1 + 64*x3) % 192)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 129, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(in_ptr3 + (4128 + 32*(((x1 + 64*x3) % 192)) + 8256*((x1 + 64*x3) // 6144) + ((((x1 + 64*x3) // 192) % 32))), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp12
    tmp16 = tl.full([1], 192, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tl.load(in_ptr3 + (8224 + 8256*((x1 + 64*x3) // 6144) + ((((x1 + 64*x3) // 192) % 32))), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp13, tmp14, tmp18)
    tmp20 = ((x0 // 64) % 192)
    tmp21 = tmp20 >= tmp10
    tmp22 = tmp20 < tmp12
    tmp23 = tl.load(in_ptr3 + (4128 + 32*(((x2 + 64*x3) % 192)) + 8256*((x2 + 64*x3) // 6144) + ((((x2 + 64*x3) // 192) % 32))), tmp22, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp20 >= tmp12
    tmp25 = tmp20 < tmp16
    tmp26 = tl.load(in_ptr3 + (8224 + 8256*(x4 // 6144) + (((x4 // 192) % 32))), tmp24, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp22, tmp23, tmp26)
    tmp28 = tmp19 - tmp27
    tmp29 = 0.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tl_math.exp(tmp30)
    tmp32 = x1
    tmp33 = x2
    tmp34 = tmp32 >= tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp31 * tmp35
    tmp37 = tmp3 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp6 * tmp36
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp38, None)
    tl.store(out_ptr2 + (x0), tmp40, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/ke/cke3urri5oiknqxdqw4rscmwzoniqnvhsucyrzxo4s5zqji3guho.py
# Topologically Sorted Source Nodes: [sub, clamp_max, exp, causal, E0, mul_4, to_18, lower, mul_14, mul_15, to_24, mul_17, mul_18, to_28], Original ATen: [aten.sub, aten.clamp_max, aten.exp, aten.ge, aten.mul, aten._to_copy, aten.gt]
# Source node to ATen node mapping:
#   E0 => mul_1
#   causal => ge
#   clamp_max => clamp_max
#   exp => exp
#   lower => gt
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_4 => mul_4
#   sub => sub
#   to_18 => convert_element_type_21
#   to_24 => convert_element_type_32
#   to_28 => convert_element_type_37
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_5, %unsqueeze_6), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%sub, 0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%clamp_max,), kwargs = {})
#   %ge : [num_users=2] = call_function[target=torch.ops.aten.ge.Tensor](args = (%unsqueeze_1, %unsqueeze_2), kwargs = {})
#   %mul_1 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %ge), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_23, %mul_1), kwargs = {})
#   %convert_element_type_21 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Tensor](args = (%unsqueeze_3, %unsqueeze_4), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %gt), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_24, %mul_14), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %gt), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_13, %mul_17), kwargs = {})
#   %convert_element_type_37 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_clamp_max_exp_ge_gt_mul_sub_8 = async_compile.triton('triton_poi_fused__to_copy_clamp_max_exp_ge_gt_mul_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_max_exp_ge_gt_mul_sub_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 6291456, 'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_max_exp_ge_gt_mul_sub_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    x2 = xindex
    tmp28 = tl.load(in_ptr1 + (y0 + 64*x2 + 4096*y1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (y0 + 64*x2 + 4096*y1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (x2 + 64*y3), xmask, eviction_policy='evict_last')
    tmp0 = (y3 % 192)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 129, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(32*(((y0 + 64*y1) % 192)) + 8256*((y0 + 64*y1) // 6144) + ((((y0 + 64*y1) // 192) % 32)), [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (tl.broadcast_to(4096 + 8256*(y3 // 6144) + (((y3 // 192) % 32)), [YBLOCK, XBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = ((x2 + 64*y1) % 192)
    tmp12 = tmp11 >= tmp1
    tmp13 = tmp11 < tmp3
    tmp14 = tl.load(in_ptr0 + (32*(((x2 + 64*y1) % 192)) + 8256*((x2 + 64*y1) // 6144) + ((((x2 + 64*y1) // 192) % 32))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp11 >= tmp3
    tmp16 = tmp11 < tmp7
    tmp17 = tl.load(in_ptr0 + (4096 + 8256*((x2 + 64*y1) // 6144) + ((((x2 + 64*y1) // 192) % 32))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp13, tmp14, tmp17)
    tmp19 = tmp10 - tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tl_math.exp(tmp21)
    tmp23 = y0
    tmp24 = x2
    tmp25 = tmp23 >= tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp22 * tmp26
    tmp29 = tmp28 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp32 = tmp23 > tmp24
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp27 * tmp33
    tmp35 = tmp31 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp38 = tmp37 * tmp34
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr1 + (x2 + 64*y3), tmp30, xmask)
    tl.store(out_ptr2 + (x2 + 64*y3), tmp36, xmask)
    tl.store(out_ptr3 + (x2 + 64*y3), tmp39, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/7q/c7qogctgautghjmqlxfc7brb64i7igs74tti4lgio4wuwic65x6i.py
# Topologically Sorted Source Nodes: [float_11, mul_23, sum_5, float_12, mul_24, d, triton_kernel_wrapper_mutation, mul_6, mul_8, add_1, sub_3, float_5, mul_10, float_6, mul_11, sum_1, mul_12, add_2, mul_13, sub_4, dk, float_7, mul_16, r0, sub_6, mul_19, dbeta], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.sub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   d => sum_6
#   dbeta => sum_2
#   dk => sub_5
#   float_11 => convert_element_type_44
#   float_12 => convert_element_type_45
#   float_5 => convert_element_type_30
#   float_6 => convert_element_type_31
#   float_7 => convert_element_type_40
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_16 => mul_16
#   mul_19 => mul_19
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_6 => mul_6
#   mul_8 => mul_8
#   r0 => add_3
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_6 => sub_6
#   sum_1 => sum_1
#   sum_5 => sum_5
#   triton_kernel_wrapper_mutation => triton_kernel_wrapper_mutation
# Graph fragment:
#   %convert_element_type_44 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_44, %bmm_7), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [-1]), kwargs = {})
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_45, %bmm_8), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_24, [-1]), kwargs = {})
#   %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 0, grid: [(192, 1, 1)], tma_descriptor_metadata: {}, kwargs: {M: %sub_8, RawG0: %view_5, G0: %view_6, G1: %view_7, Bterm: %sub_7, Dterm: %sum_6, Sterm: %sum_3, Out: %permute_26}})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_10, %bmm_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_10, 0.08838834764831845), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_8), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %bmm_11), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %convert_element_type_30), kwargs = {})
#   %convert_element_type_31 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.float32), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, %bmm), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_11, [-1]), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_12), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_3, %mul_12), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %bmm_7), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %mul_13), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_4, %bmm_12), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_14, %bmm_5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %bmm_14), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_40, %add_3), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %bmm), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [-1]), kwargs = {})
triton_red_fused__to_copy_add_mul_sub_sum_9 = async_compile.triton('triton_red_fused__to_copy_add_mul_sub_sum_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*bf16', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*bf16', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sub_sum_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 29, 'num_reduction': 4, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3686400, 'r0_': 72351744}}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sub_sum_9(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr2, out_ptr3, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = (x2 % 192)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 129, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*(((x0 + 64*x1) % 192)) + 8256*((x0 + 64*x1) // 6144) + ((((x0 + 64*x1) // 192) % 32))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (4096 + 8256*(x2 // 6144) + (((x2 // 192) % 32))), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp45 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp11 = tl.load(in_ptr1 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr4 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr5 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr6 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr7 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr8 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr9 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp61 = tl.load(in_out_ptr0 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr3 + (129*r0_3 + 16512*((((r0_3 + 128*x0 + 8192*x1) // 24576) % 32)) + 1056768*((r0_3 + 128*x0 + 8192*x1) // 786432) + (((x0 + 64*x1) % 192))), r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.load(in_ptr0 + (tl.broadcast_to(32*(((x0 + 64*x1) % 192)) + 8256*((x0 + 64*x1) // 6144) + ((((x0 + 64*x1) // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr0 + (tl.broadcast_to(4096 + 8256*(x2 // 6144) + (((x2 // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp6, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.where(tmp4, tmp20, tmp21)
        tmp23 = tl_math.exp(tmp22)
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = tmp19 - tmp27
        tmp29 = tmp28 * tmp13
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask, tmp32, _tmp31)
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp34 * tmp35
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask, tmp39, _tmp38)
        tmp41 = tmp40.to(tl.float32)
        tmp43 = tmp41 * tmp42
        tmp44 = tl.broadcast_to(tmp43, [XBLOCK, R0_BLOCK])
        tmp46 = _tmp45 + tmp44
        _tmp45 = tl.where(r0_mask, tmp46, _tmp45)
        tmp47 = ((63 + 64*x1) % 192)
        tmp48 = tmp47 >= tmp1
        tmp49 = tmp47 < tmp3
        tmp50 = tl.load(in_ptr0 + (tl.broadcast_to(4128 + 32*(((63 + 64*x1) % 192)) + 8256*((63 + 64*x1) // 6144) + ((((63 + 64*x1) // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tmp47 >= tmp3
        tmp52 = tmp47 < tmp7
        tmp53 = tl.load(in_ptr0 + (tl.broadcast_to(8224 + 8256*((63 + 64*x1) // 6144) + ((((63 + 64*x1) // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp51, eviction_policy='evict_last', other=0.0)
        tmp54 = tl.where(tmp49, tmp50, tmp53)
        tmp55 = tl.load(in_ptr0 + (tl.broadcast_to(4128 + 32*(((x0 + 64*x1) % 192)) + 8256*((x0 + 64*x1) // 6144) + ((((x0 + 64*x1) // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr0 + (tl.broadcast_to(8224 + 8256*(x2 // 6144) + (((x2 // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp6, eviction_policy='evict_last', other=0.0)
        tmp57 = tl.where(tmp4, tmp55, tmp56)
        tmp58 = tmp54 - tmp57
        tmp59 = tl_math.exp(tmp58)
        tmp60 = tmp59 * tmp35
        tmp62 = 0.08838834764831845
        tmp63 = tmp61 * tmp62
        tmp64 = tmp60 + tmp63
        tl.store(in_out_ptr0 + (r0_3 + 128*x2), tmp64, r0_mask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tmp45 = tl.sum(_tmp45, 1)[:, None]
    tl.store(out_ptr2 + (x2), tmp31, None)
    tl.store(out_ptr3 + (x2), tmp38, None)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp65 = tl.load(in_out_ptr0 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp66 = tl.load(in_ptr10 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.load(in_ptr8 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp87 = tl.load(in_ptr9 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp90 = tl.load(in_ptr12 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp67 = tmp65 - tmp66
        tmp68 = (x2 % 192)
        tmp69 = tl.full([1, 1], 129, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tl.load(in_ptr11 + (tl.broadcast_to(4128 + 32*((x2 % 192)) + 8256*(x2 // 6144) + (((x2 // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp70, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp72 = tmp71.to(tl.float32)
        tmp74 = tmp73.to(tl.float32)
        tmp75 = tmp72 * tmp74
        tmp76 = tmp75 * tmp16
        tmp77 = tmp67 + tmp76
        tmp78 = tl.full([1, 1], 0, tl.int64)
        tmp79 = tmp68 >= tmp78
        tmp80 = tl.load(in_ptr0 + (tl.broadcast_to(32*(((x0 + 64*x1) % 192)) + 8256*((x0 + 64*x1) // 6144) + ((((x0 + 64*x1) // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp70, eviction_policy='evict_last', other=0.0)
        tmp81 = tmp68 >= tmp69
        tmp82 = tl.full([1, 1], 192, tl.int64)
        tmp83 = tmp68 < tmp82
        tmp84 = tl.load(in_ptr0 + (tl.broadcast_to(4096 + 8256*(x2 // 6144) + (((x2 // 192) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp85 = tl.where(tmp70, tmp80, tmp84)
        tmp86 = tl_math.exp(tmp85)
        tmp88 = tmp86 * tmp87
        tmp89 = tmp77 - tmp88
        tmp91 = tmp89 - tmp90
        tl.store(in_out_ptr0 + (r0_3 + 128*x2), tmp91, r0_mask)
    tmp92 = tl.load(in_out_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp93 = 0.08838834764831845
    tmp94 = tmp92 * tmp93
    tmp95 = tmp94 - tmp45
    tmp96 = (x2 % 192)
    tmp97 = tl.full([1, 1], 0, tl.int64)
    tmp98 = tmp96 >= tmp97
    tmp99 = tl.full([1, 1], 129, tl.int64)
    tmp100 = tmp96 < tmp99
    tmp101 = tl.load(in_ptr0 + (4128 + 32*(((x0 + 64*x1) % 192)) + 8256*((x0 + 64*x1) // 6144) + ((((x0 + 64*x1) // 192) % 32))), tmp100, eviction_policy='evict_last', other=0.0)
    tmp102 = tmp96 >= tmp99
    tmp103 = tl.full([1, 1], 192, tl.int64)
    tmp104 = tmp96 < tmp103
    tmp105 = tl.load(in_ptr0 + (8224 + 8256*(x2 // 6144) + (((x2 // 192) % 32))), tmp102, eviction_policy='evict_last', other=0.0)
    tmp106 = tl.where(tmp100, tmp101, tmp105)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2), tmp95, None)
    tl.store(out_ptr5 + (x2), tmp106, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/cc/cccdm77dcl667mmmyxl6rwngynt5e7ipit2vweduxphaab6tjlqw.py
# Topologically Sorted Source Nodes: [float_8, float_9, mul_20, S], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   S => sum_3
#   float_8 => convert_element_type_41
#   float_9 => convert_element_type_42
#   mul_20 => mul_20
# Graph fragment:
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.float32), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_41, %convert_element_type_42), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [-1, -2]), kwargs = {})
triton_per_fused__to_copy_mul_sum_10 = async_compile.triton('triton_per_fused__to_copy_mul_sum_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mul_sum_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1536, 'r0_': 384}}
)
@triton.jit
def triton_per_fused__to_copy_mul_sum_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 192
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/3m/c3mb5hjkjfjwo5nn57qri3rhraub6kpylpgzzbahckeu5zynorzx.py
# Topologically Sorted Source Nodes: [triton_kernel_wrapper_mutation], Original ATen: []
# Source node to ATen node mapping:
#   triton_kernel_wrapper_mutation => triton_kernel_wrapper_mutation
# Graph fragment:
#   %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 0, grid: [(192, 1, 1)], tma_descriptor_metadata: {}, kwargs: {M: %sub_8, RawG0: %view_5, G0: %view_6, G1: %view_7, Bterm: %sub_7, Dterm: %sum_6, Sterm: %sum_3, Out: %permute_26}})
triton_poi_fused_11 = async_compile.triton('triton_poi_fused_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = (x0 % 192)
    tmp1 = tl.full([1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (32*((x0 % 192)) + 8256*(x0 // 6144) + (((x0 // 192) % 32))), tmp2, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# Original path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/finite_fla_gpu.py:23
_finite_decay_scan_0 = async_compile.triton('_finite_decay_scan', '''

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 1}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_finite_decay_scan_0', 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'M': '*fp32', 'RawG0': '*fp32', 'G0': '*fp32', 'G1': '*fp32', 'Bterm': '*fp32', 'Dterm': '*fp32', 'Sterm': '*fp32', 'Out': '*fp32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'C': 64}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _finite_decay_scan(M, RawG0, G0, G1, Bterm, Dterm, Sterm, Out, C: tl.constexpr):
    """One [64,64] chunk/head; native state matrices never enter this scan."""
    block = tl.program_id(0)
    i = tl.arange(0, C)
    t = tl.arange(0, C)
    prev = tl.maximum(t - 1, 0)
    # Rows are future source i; columns are prefix cut t. Shift before scanning
    # so the inclusive affine scan yields the exclusive past P[t-1,i].
    m = tl.load(M + block*C*C + prev[None, :]*C + i[:, None], t[None, :] > 0, 0)
    a = tl.exp(tl.load(RawG0 + block*C + prev))
    a = tl.where(t > 0, a, 1.)
    a = tl.broadcast_to(a[None, :], (C, C))
    _, past = tl.associative_scan((a, m), 1, _affine_compose)
    g0i = tl.load(G0 + block*C + i)
    g0prev = tl.where(t > 0, tl.load(G0 + block*C + prev), 0.)
    g1i = tl.load(G1 + block*C + i)
    g1t = tl.load(G1 + block*C + t)
    e1 = tl.exp(tl.minimum(g1i[:, None] - g1t[None, :], 0.))
    e1 = tl.where(i[:, None] >= t[None, :], e1, 0.)
    ep = tl.exp(tl.minimum(g0prev[None, :] - g0i[:, None], 0.))
    ep = tl.where(i[:, None] < t[None, :], ep, 0.)
    b = tl.load(Bterm + block*C + i)
    d = tl.load(Dterm + block*C + i)
    s = tl.load(Sterm + block)
    end = tl.load(G1 + block*C + C - 1)
    out = (tl.exp(end - g1t) * (tl.exp(g0prev) * s + tl.sum(ep*d[:, None], 0))
           + tl.exp(g0prev) * tl.sum(e1*b[:, None], 0) + tl.sum(e1*past, 0))
    tl.store(Out + block*C + t, out)

@triton.jit
def _affine_compose(a_left, v_left, a_right, v_right):
    return a_right * a_left, v_right + a_right * v_left
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/lp/clpn73zajubpeclyhbz6ihvcrqajj7c4shy53euezwzzpgcgzyy7.py
# Topologically Sorted Source Nodes: [contiguous_15], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_15 => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 8454144, 'x': 16908288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 16512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 128)
    x3 = xindex // 128
    y0 = (yindex % 32)
    y1 = yindex // 32
    y4 = yindex
    tmp12 = tl.load(in_ptr1 + (x2 + 128*((x3 % 64)) + 8192*((x2 + 128*x3) // 8192) + 24576*y4), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x2 + 128*((x3 % 64)) + 8192*((x2 + 128*x3) // 8192) + 24576*y4), xmask & ymask, eviction_policy='evict_last')
    tmp0 = 64*((x2 + 128*x3) // 8192) + ((x3 % 64))
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 129, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 32*(64*((x2 + 128*x3) // 8192) + ((x3 % 64))) + 8256*y1 + 8256*(triton_helpers.div_floor_integer(64*((x2 + 128*x3) // 8192) + 192*y0 + ((x3 % 64)),  6144)) + (triton_helpers.div_floor_integer(64*((x2 + 128*x3) // 8192) + ((x3 % 64)),  192))), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (4096 + y0 + 8256*y1 + 8256*(triton_helpers.div_floor_integer(64*((x2 + 128*x3) // 8192) + 192*y0 + ((x3 % 64)),  6144)) + (triton_helpers.div_floor_integer(64*((x2 + 128*x3) // 8192) + ((x3 % 64)),  192))), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tl_math.exp(tmp10)
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.08838834764831845
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2 + 128*y0 + 4096*x3 + 528384*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/le/clexx56qlzjaq4zvnujtlxswxaudtebkwnfmbllrs56rlisoboqd.py
# Topologically Sorted Source Nodes: [contiguous_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_16 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_29,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_13 = async_compile.triton('triton_poi_fused_clone_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12681216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1056768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 32)
    x2 = ((xindex // 4096) % 129)
    x3 = xindex // 528384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 24576*x1 + 786432*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/i3/ci3a6e7hlmurovbze32u3mvq4wmh2uhoomiq4o3iulyttqnabkwn.py
# Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_17 => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_30,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_14 = async_compile.triton('triton_poi_fused_clone_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2113536, 'x': 12681216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8256
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y1 = ((yindex // 32) % 129)
    y0 = (yindex % 32)
    y2 = yindex // 4128
    y4 = yindex
    tmp5 = tl.load(in_ptr1 + (x3 + 128*((y1 % 64)) + 8192*((x3 + 128*y1) // 8192) + 24576*y0 + 786432*y2), xmask & ymask, eviction_policy='evict_last')
    tmp0 = 64*((x3 + 128*y1) // 8192) + ((y1 % 64))
    tmp1 = tl.full([1, 1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4128 + y0 + 32*((y1 % 64)) + 2048*((x3 + 128*y1) // 8192) + 8256*y2 + 8256*(triton_helpers.div_floor_integer(64*((x3 + 128*y1) // 8192) + 192*y0 + ((y1 % 64)),  6144)) + (triton_helpers.div_floor_integer(64*((x3 + 128*y1) // 8192) + ((y1 % 64)),  192))), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x3 + 128*y4), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/sh/cshjzmthzzp7ruwsa47gs6abodrkldjoptnds23pcxlbpqnh3smb.py
# Topologically Sorted Source Nodes: [contiguous_18], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_18 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_31,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_15 = async_compile.triton('triton_poi_fused_clone_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 33024, 'x': 66048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 258
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 129)
    y1 = yindex // 129
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 6144*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 32*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1/inductor_cache/y7/cy7bmnmzi77yqzax42wp3zgd4ucy6au2k26i7tp6ft7dt4lueyl6.py
# Topologically Sorted Source Nodes: [contiguous_19, contiguous_20], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_19 => clone_6
#   contiguous_20 => clone_7
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_33,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 198144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 129)
    x2 = xindex // 4128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 192*x0 + 6144*x2), xmask, eviction_policy='evict_last')
    tmp1 = x1
    tmp2 = tl.full([1], 129, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x0 + 32*x1 + 8256*x2 + 8256*(triton_helpers.div_floor_integer(64*(x1 // 64) + 192*x0 + ((x1 % 64)),  6144)) + (triton_helpers.div_floor_integer(64*(x1 // 64) + ((x1 % 64)),  192))), tmp3 & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (4128 + x0 + 32*x1 + 8256*x2 + 8256*(triton_helpers.div_floor_integer(64*(x1 // 64) + 192*x0 + ((x1 % 64)),  6144)) + (triton_helpers.div_floor_integer(64*(x1 // 64) + ((x1 % 64)),  192))), tmp3 & xmask, other=0.0)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp0 * tmp7
    tmp9 = tmp5 - tmp4
    tmp10 = tl_math.abs(tmp9)
    tmp11 = 0.0
    tmp12 = tmp10 == tmp11
    tmp13 = -tmp10
    tmp14 = libdevice.expm1(tmp13)
    tmp15 = -tmp14
    tmp16 = 1.0
    tmp17 = tl.where(tmp12, tmp16, tmp10)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tl.where(tmp12, tmp16, tmp18)
    tmp20 = tmp8 * tmp19
    tl.store(out_ptr0 + (x3), tmp0, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 129, 32, 128), (528384, 1, 16512, 129))
    assert_size_stride(arg1_1, (4, 129, 32, 128), (528384, 1, 16512, 129))
    assert_size_stride(arg2_1, (4, 129, 32, 128), (528384, 1, 16512, 129))
    assert_size_stride(arg3_1, (4, 129, 32, 128), (528384, 1, 16512, 129))
    assert_size_stride(arg4_1, (4, 129, 32), (4128, 32, 1))
    assert_size_stride(arg5_1, (4, 129, 32), (4128, 32, 1))
    assert_size_stride(arg6_1, (4, 129, 32), (4128, 32, 1))
    assert_size_stride(arg7_1, (4, 129, 32, 64), (264192, 1, 8256, 129))
    assert_size_stride(arg8_1, (2, 129, 32, 128), (528384, 4096, 128, 1))
    assert_size_stride(arg9_1, (2, 129, 32, 128), (528384, 4096, 128, 1))
    assert_size_stride(arg10_1, (4, 3, 32, 128, 128), (1572864, 524288, 16384, 128, 1))
    assert_size_stride(arg11_1, (2, 3, 32, 128, 128), (1572864, 524288, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 32, 192, 128), (786432, 24576, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [a_3], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(arg2_1, buf0, 12288, 128, stream=stream0)
        buf13 = empty_strided_cuda((2, 32, 3, 128, 128), (1572864, 49152, 16384, 128, 1), torch.bfloat16)
        buf19 = empty_strided_cuda((2, 32, 3, 128, 128), (1572864, 49152, 16384, 128, 1), torch.bfloat16)
        buf22 = empty_strided_cuda((192, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_12, reshape_13, float_8, float_9, mul_20, S], Original ATen: [aten.clone, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_clone_mul_sum_1.run(arg10_1, arg11_1, buf13, buf19, buf22, 384, 8192, stream=stream0)
        del arg10_1
        del arg11_1
        buf4 = empty_strided_cuda((2, 32, 192, 128), (786432, 24576, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [a_21], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2.run(arg8_1, buf4, 1572864, stream=stream0)
        del arg8_1
        buf14 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ZHt], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf4, (192, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf13, (192, 128, 128), (16384, 1, 128), 0), out_dtype=torch.float32, out=buf14)
        buf1 = empty_strided_cuda((2, 32, 192, 128), (786432, 24576, 128, 1), torch.bfloat16)
        buf15 = empty_strided_cuda((192, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [a_1, float_10, mul_21, sum_4], Original ATen: [aten.constant_pad_nd, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_constant_pad_nd_mul_sum_3.run(arg0_1, buf14, buf1, buf15, 12288, 128, stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((192, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [K0Q], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf0, (192, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf1, (192, 128, 64), (8192, 1, 128), 0), out_dtype=torch.float32, out=buf2)
        buf3 = empty_strided_cuda((2, 32, 192, 128), (786432, 24576, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [a_9], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(arg3_1, buf3, 12288, 128, stream=stream0)
        del arg3_1
        buf5 = empty_strided_cuda((192, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [UZ], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf3, (192, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf4, (192, 128, 64), (8192, 1, 128), 0), out_dtype=torch.float32, out=buf5)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [a_5], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_4.run(arg2_1, buf6, 12288, 128, stream=stream0)
        del arg2_1
        buf7 = empty_strided_cuda((192, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [K0K1], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf0, (192, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf6, (192, 128, 64), (8192, 1, 128), 0), out_dtype=torch.float32, out=buf7)
        buf8 = empty_strided_cuda((2, 32, 192, 64), (393216, 12288, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [a_19], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_5.run(arg7_1, buf8, 12288, 64, stream=stream0)
        del arg7_1
        buf9 = empty_strided_cuda((2, 32, 192, 128), (786432, 24576, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [a_23], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2.run(arg9_1, buf9, 1572864, stream=stream0)
        del arg9_1
        buf10 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [L], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf8, (192, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf9, (192, 64, 128), (8192, 128, 1), 0), out_dtype=torch.float32, out=buf10)
        buf11 = reinterpret_tensor(buf9, (192, 128, 64), (8192, 1, 128), 0); del buf9  # reuse
        buf16 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [W, to_5, to_14], Original ATen: [aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_6.run(arg6_1, buf10, buf11, buf16, 1572864, stream=stream0)
        buf12 = empty_strided_cuda((192, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to_5, UW], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf3, (192, 64, 128), (8192, 128, 1), 0), buf11, out_dtype=torch.float32, out=buf12)
        del buf11
        buf17 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [W, to_14, WHt], Original ATen: [aten.mul, aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf16, reinterpret_tensor(buf13, (192, 128, 128), (16384, 1, 128), 0), out_dtype=torch.float32, out=buf17)
        del buf16
        buf20 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [UDt], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf3, (192, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf19, (192, 128, 128), (16384, 1, 128), 0), out_dtype=torch.float32, out=buf20)
        del buf19
        buf25 = buf2; del buf2  # reuse
        buf36 = reinterpret_tensor(buf8, (192, 64, 64), (4096, 64, 1), 0); del buf8  # reuse
        buf39 = empty_strided_cuda((192, 64, 64), (4096, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [triton_kernel_wrapper_mutation, sub_1, clamp_max_1, exp_1, E1, mul_7, to_20, mul_9, to_22], Original ATen: [aten.sub, aten.clamp_max, aten.exp, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_max_exp_mul_sub_7.run(buf25, buf5, buf7, buf12, arg5_1, buf36, buf39, 786432, stream=stream0)
        buf37 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_7, to_20, bmm_10], Original ATen: [aten.mul, aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf36, reinterpret_tensor(buf1, (192, 64, 128), (8192, 128, 1), 0), out_dtype=torch.float32, out=buf37)
        del buf1
        buf40 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_9, to_22, bmm_11], Original ATen: [aten.mul, aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf39, reinterpret_tensor(buf6, (192, 64, 128), (8192, 128, 1), 0), out_dtype=torch.float32, out=buf40)
        buf48 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [bmm_13], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf0, (192, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf0, (192, 128, 64), (8192, 1, 128), 0), out_dtype=torch.float32, out=buf48)
        buf32 = buf39; del buf39  # reuse
        buf42 = buf36; del buf36  # reuse
        buf49 = empty_strided_cuda((192, 64, 64), (4096, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sub, clamp_max, exp, causal, E0, mul_4, to_18, lower, mul_14, mul_15, to_24, mul_17, mul_18, to_28], Original ATen: [aten.sub, aten.clamp_max, aten.exp, aten.ge, aten.mul, aten._to_copy, aten.gt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_max_exp_ge_gt_mul_sub_8.run(arg5_1, buf5, buf12, buf48, buf32, buf42, buf49, 12288, 64, stream=stream0)
        del buf12
        del buf48
        del buf5
        buf43 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lower, mul_14, mul_15, to_24, bmm_12], Original ATen: [aten.gt, aten.mul, aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf42, reinterpret_tensor(buf0, (192, 64, 128), (8192, 128, 1), 0), out_dtype=torch.float32, out=buf43)
        del buf42
        buf47 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [K0H], Original ATen: [aten.bmm]
        extern_kernels.bmm_dtype(reinterpret_tensor(buf0, (192, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf13, (192, 128, 128), (16384, 128, 1), 0), out_dtype=torch.float32, out=buf47)
        del buf13
        buf50 = empty_strided_cuda((192, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lower, mul_17, mul_18, to_28, bmm_14], Original ATen: [aten.gt, aten.mul, aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf49, reinterpret_tensor(buf3, (192, 64, 128), (8192, 128, 1), 0), out_dtype=torch.float32, out=buf50)
        del buf49
        buf27 = empty_strided_cuda((192, 64), (64, 1), torch.float32)
        buf51 = empty_strided_cuda((192, 64), (64, 1), torch.float32)
        buf21 = empty_strided_cuda((192, 64), (64, 1), torch.float32)
        buf38 = buf37; del buf37  # reuse
        buf44 = buf38; del buf38  # reuse
        buf29 = buf15; del buf15  # reuse
        buf28 = empty_strided_cuda((192, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_11, mul_23, sum_5, float_12, mul_24, d, triton_kernel_wrapper_mutation, mul_6, mul_8, add_1, sub_3, float_5, mul_10, float_6, mul_11, sum_1, mul_12, add_2, mul_13, sub_4, dk, float_7, mul_16, r0, sub_6, mul_19, dbeta], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.sub]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sub_sum_9.run(buf44, buf29, arg5_1, buf3, buf10, arg1_1, buf47, buf50, buf0, buf20, buf6, buf17, buf40, arg6_1, buf43, buf27, buf51, buf21, buf28, 12288, 128, stream=stream0)
        del arg1_1
        del buf17
        del buf20
        del buf3
        del buf40
        del buf43
        del buf47
        del buf6
        buf23 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [float_8, float_9, mul_20, S], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mul_sum_10.run(buf22, buf23, 192, 2, stream=stream0)
        del buf22
        buf24 = empty_strided_cuda((192, 64), (64, 1), torch.float32)
        buf26 = empty_strided_cuda((192, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [triton_kernel_wrapper_mutation], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(arg4_1, buf26, 12288, stream=stream0)
        # Topologically Sorted Source Nodes: [triton_kernel_wrapper_mutation], Original ATen: []
        stream0 = get_raw_stream(0)
        _finite_decay_scan_0.run(buf25, buf26, buf27, buf28, buf29, buf21, buf23, buf24, 64, 192, 1, 1, stream=stream0)
        del buf21
        del buf23
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        buf33 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [mul_4, to_18, bmm_9], Original ATen: [aten.mul, aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf32, reinterpret_tensor(buf0, (192, 64, 128), (8192, 128, 1), 0), out_dtype=torch.float32, out=buf33)
        del buf0
        del buf32
        buf34 = empty_strided_cuda((2, 129, 32, 128), (528384, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_15], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(arg5_1, buf14, buf33, buf34, 64, 16512, stream=stream0)
        del arg5_1
        del buf14
        del buf33
        buf45 = empty_strided_cuda((2, 129, 32, 128), (528384, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_16], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf44, buf45, 1056768, stream=stream0)
        del buf44
        buf46 = empty_strided_cuda((2, 129, 32, 128), (528384, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_14.run(arg6_1, buf10, buf46, 8256, 128, stream=stream0)
        del arg6_1
        del buf10
        buf52 = empty_strided_cuda((2, 129, 32), (4128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_18], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf51, buf52, 258, 32, stream=stream0)
        del buf51
        buf53 = empty_strided_cuda((2, 129, 32), (4128, 32, 1), torch.float32)
        buf54 = empty_strided_cuda((2, 129, 32), (4128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_19, contiguous_20], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf24, arg4_1, buf53, buf54, 8256, stream=stream0)
        del arg4_1
        del buf24
    return (buf34, buf45, buf46, buf52, buf53, buf54, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 129, 32, 128), (528384, 1, 16512, 129), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, 129, 32, 128), (528384, 1, 16512, 129), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4, 129, 32, 128), (528384, 1, 16512, 129), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4, 129, 32, 128), (528384, 1, 16512, 129), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((4, 129, 32), (4128, 32, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4, 129, 32), (4128, 32, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((4, 129, 32), (4128, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((4, 129, 32, 64), (264192, 1, 8256, 129), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((2, 129, 32, 128), (528384, 4096, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((2, 129, 32, 128), (528384, 4096, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((4, 3, 32, 128, 128), (1572864, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((2, 3, 32, 128, 128), (1572864, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
