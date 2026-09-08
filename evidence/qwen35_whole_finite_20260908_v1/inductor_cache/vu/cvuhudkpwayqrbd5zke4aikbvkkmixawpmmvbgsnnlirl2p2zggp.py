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


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/js/cjs3bbqyn6td4zptpekpp65ogt45az6qiqjible4c4l3vejxo3ci.py
# Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   z0 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, -1), kwargs = {})
triton_red_fused__to_copy_0 = async_compile.triton('triton_red_fused__to_copy_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3776, 'r0_': 29301760}}
)
@triton.jit
def triton_red_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 472
    r0_numel = 31040
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 31040*x0 + 496640*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/oa/coakz5zch5xge2qsuwa2xdrkl4cp74adqdjsa3kfwr5obxm7srwm.py
# Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   z0 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, -1), kwargs = {})
triton_per_fused__to_copy_1 = async_compile.triton('triton_per_fused__to_copy_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 472, 'r0_': 1888}}
)
@triton.jit
def triton_per_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 59
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/ac/caczpzvcrwydglpykbblagl7nac52ykxkdxcfb7muknw2evrjmrx.py
# Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   z0 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, -1), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, -1), kwargs = {})
triton_red_fused__to_copy_2 = async_compile.triton('triton_red_fused__to_copy_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7552, 'r0_': 29301760}}
)
@triton.jit
def triton_red_fused__to_copy_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 472
    r0_numel = 31040
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 31040*x0 + 496640*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp4 = tl_math.exp(tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp8 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/qb/cqbjoxyc4wbnqij6km5horahmxp5yqzvtf4o52ig5zhyhlzazk4s.py
# Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   z0 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, -1), kwargs = {})
triton_per_fused__to_copy_3 = async_compile.triton('triton_per_fused__to_copy_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 472, 'r0_': 1888}}
)
@triton.jit
def triton_per_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 59
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/d7/cd7dvkcot53nuasajkgcllytlxc6kdpockfskfeyiqvrqzh4gztq.py
# Topologically Sorted Source Nodes: [z0, lp0, z1, lp1, isfinite_2, isfinite_3, eq, same_mask], Original ATen: [aten._to_copy, aten._log_softmax, aten.eq, aten.abs, aten.ne, aten.mul, aten.all]
# Source node to ATen node mapping:
#   eq => eq_4
#   isfinite_2 => abs_3, eq_2, mul_2, ne_2
#   isfinite_3 => abs_4, eq_3, mul_3, ne_3
#   lp0 => log, sub_1
#   lp1 => log_1, sub_3
#   same_mask => any_1, logical_not
#   z0 => convert_element_type
#   z1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %sub_tensor_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_6), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_7,), kwargs = {})
#   %sub_1 : [num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_3, %log), kwargs = {})
#   %convert_element_type_1 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %sub_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %getitem_4), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_5,), kwargs = {})
#   %sub_3 : [num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_2, %log_1), kwargs = {})
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%sub_1, %sub_1), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_3, inf), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_2, %ne_2), kwargs = {})
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%sub_3, %sub_3), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_4, inf), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_3, %ne_3), kwargs = {})
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_4,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not,), kwargs = {})
triton_red_fused__log_softmax__to_copy_abs_all_eq_mul_ne_4 = async_compile.triton('triton_red_fused__log_softmax__to_copy_abs_all_eq_mul_ne_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_abs_all_eq_mul_ne_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 896, 'r0_': 58603776}}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_abs_all_eq_mul_ne_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 448
    r0_numel = 32703
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + 32703*x0
        tmp1 = tl.full([1, 1], 14650880, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (496640*((((r0_1 + 32703*x0) // 248320) % 59)) + (((r0_1 + 32703*x0) % 248320))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + ((((r0_1 + 32703*x0) // 248320) % 59)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr2 + ((((r0_1 + 32703*x0) // 248320) % 59)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl_math.log(tmp7)
        tmp9 = tmp6 - tmp8
        tmp10 = tmp9 == tmp9
        tmp11 = tl_math.abs(tmp9)
        tmp12 = float("inf")
        tmp13 = tmp11 != tmp12
        tmp14 = tmp10 & tmp13
        tmp15 = tl.load(in_ptr3 + (496640*((((r0_1 + 32703*x0) // 248320) % 59)) + (((r0_1 + 32703*x0) % 248320))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.load(in_ptr4 + ((((r0_1 + 32703*x0) // 248320) % 59)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tl.load(in_ptr5 + ((((r0_1 + 32703*x0) // 248320) % 59)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl_math.log(tmp19)
        tmp21 = tmp18 - tmp20
        tmp22 = tmp21 == tmp21
        tmp23 = tl_math.abs(tmp21)
        tmp24 = tmp23 != tmp12
        tmp25 = tmp22 & tmp24
        tmp26 = tmp14 == tmp25
        tmp27 = tmp26 == 0
        tmp28 = tl.full(tmp27.shape, False, tmp27.dtype)
        tmp29 = tl.where(tmp2, tmp27, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 | tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
    tmp31 = triton_helpers.any(_tmp31.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/er/cerc3cadw77qnexa32c2enhi34zql2yl6h2bm35gsnmlq5ffhd22.py
# Topologically Sorted Source Nodes: [z0, lp0, z1, lp1, isfinite_2, isfinite_3, eq, same_mask], Original ATen: [aten._to_copy, aten._log_softmax, aten.eq, aten.abs, aten.ne, aten.mul, aten.all]
# Source node to ATen node mapping:
#   eq => eq_4
#   isfinite_2 => abs_3, eq_2, mul_2, ne_2
#   isfinite_3 => abs_4, eq_3, mul_3, ne_3
#   lp0 => log, sub_1
#   lp1 => log_1, sub_3
#   same_mask => any_1, logical_not
#   z0 => convert_element_type
#   z1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %sub_tensor_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_6), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_7,), kwargs = {})
#   %sub_1 : [num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_3, %log), kwargs = {})
#   %convert_element_type_1 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %sub_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %getitem_4), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_5,), kwargs = {})
#   %sub_3 : [num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_2, %log_1), kwargs = {})
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%sub_1, %sub_1), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_3, inf), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_2, %ne_2), kwargs = {})
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%sub_3, %sub_3), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_4, inf), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_3, %ne_3), kwargs = {})
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_4,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not,), kwargs = {})
triton_per_fused__log_softmax__to_copy_abs_all_eq_mul_ne_5 = async_compile.triton('triton_per_fused__log_softmax__to_copy_abs_all_eq_mul_ne_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=(2,))]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy_abs_all_eq_mul_ne_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 448}}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy_abs_all_eq_mul_ne_5(in_ptr0, out_ptr0, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 448
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0).to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, False)
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp3, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/ij/cijtkp7dcewmbcq756brw4chawtdiznnpgghodswimsazctr7mc4.py
# Topologically Sorted Source Nodes: [z0, lp0, z1, lp1, maximum, exp, isfinite, isfinite_1, valid, sub, abs_1, zeros_like, distance, eq_1, ones_like_1, neg, expm1, neg_1, gt, ones_like, denominator, ratio, ratio_1, mul, zeros_like_1, mean, sum_2, sum_1], Original ATen: [aten._to_copy, aten._log_softmax, aten.maximum, aten.exp, aten.eq, aten.abs, aten.ne, aten.mul, aten.bitwise_and, aten.sub, aten.zeros_like, aten.where, aten.ones_like, aten.neg, aten.expm1, aten.gt, aten.div, aten.sum]
# Source node to ATen node mapping:
#   abs_1 => abs_5
#   denominator => where_1
#   distance => where
#   eq_1 => eq_5
#   exp => exp_2
#   expm1 => expm1
#   gt => gt
#   isfinite => abs_1, eq, mul, ne
#   isfinite_1 => abs_2, eq_1, mul_1, ne_1
#   lp0 => log, sub_1
#   lp1 => log_1, sub_3
#   maximum => maximum
#   mean => where_3
#   mul => mul_4
#   neg => neg
#   neg_1 => neg_1
#   ones_like => full_default_1
#   ones_like_1 => full_default_2
#   ratio => div
#   ratio_1 => where_2
#   sub => sub_4
#   sum_1 => sum_3
#   sum_2 => sum_4
#   valid => bitwise_and
#   z0 => convert_element_type
#   z1 => convert_element_type_1
#   zeros_like => full_default
#   zeros_like_1 => full_default_3
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %sub_tensor_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_6), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_7,), kwargs = {})
#   %sub_1 : [num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_3, %log), kwargs = {})
#   %convert_element_type_1 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %sub_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %getitem_4), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_5,), kwargs = {})
#   %sub_3 : [num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_2, %log_1), kwargs = {})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%sub_1, %sub_3), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%maximum,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%sub_1, %sub_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_1, inf), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq, %ne), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%sub_3, %sub_3), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_2, inf), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_1, %ne_1), kwargs = {})
#   %bitwise_and : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %sub_1), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([59, 248320], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %abs_5, %full_default), kwargs = {})
#   %eq_5 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%where, 0), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([59, 248320], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where,), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%neg,), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%expm1,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([59, 248320], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %where, %full_default_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg_1, %where_1), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_5, %full_default_2, %div), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_2, %where_2), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([59, 248320], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=6] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %mul_4, %full_default_3), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_3, [-1], True), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_3, [-1]), kwargs = {})
triton_red_fused__log_softmax__to_copy_abs_bitwise_and_div_eq_exp_expm1_gt_maximum_mul_ne_neg_ones_like_sub_sum_where_zeros_like_6 = async_compile.triton('triton_red_fused__log_softmax__to_copy_abs_bitwise_and_div_eq_exp_expm1_gt_maximum_mul_ne_neg_ones_like_sub_sum_where_zeros_like_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_abs_bitwise_and_div_eq_exp_expm1_gt_maximum_mul_ne_neg_ones_like_sub_sum_where_zeros_like_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7552, 'r0_': 175810560}}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_abs_bitwise_and_div_eq_exp_expm1_gt_maximum_mul_ne_neg_ones_like_sub_sum_where_zeros_like_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 472
    r0_numel = 31040
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    x3 = xindex
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 31040*x0 + 496640*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (r0_2 + 31040*x0 + 496640*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp5 = tl_math.log(tmp4)
        tmp6 = tmp3 - tmp5
        tmp7 = tmp6 == tmp6
        tmp8 = tl_math.abs(tmp6)
        tmp9 = float("inf")
        tmp10 = tmp8 != tmp9
        tmp11 = tmp7 & tmp10
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp17 = tl_math.log(tmp16)
        tmp18 = tmp15 - tmp17
        tmp19 = tmp18 == tmp18
        tmp20 = tl_math.abs(tmp18)
        tmp21 = tmp20 != tmp9
        tmp22 = tmp19 & tmp21
        tmp23 = tmp11 & tmp22
        tmp24 = tmp18 - tmp6
        tmp25 = tl_math.abs(tmp24)
        tmp26 = 0.0
        tmp27 = tl.where(tmp23, tmp25, tmp26)
        tmp28 = triton_helpers.maximum(tmp6, tmp18)
        tmp29 = tl_math.exp(tmp28)
        tmp30 = tmp27 == tmp26
        tmp31 = -tmp27
        tmp32 = libdevice.expm1(tmp31)
        tmp33 = -tmp32
        tmp34 = tmp27 > tmp26
        tmp35 = 1.0
        tmp36 = tl.where(tmp34, tmp27, tmp35)
        tmp37 = (tmp33 / tmp36)
        tmp38 = tl.where(tmp30, tmp35, tmp37)
        tmp39 = tmp29 * tmp38
        tmp40 = tl.where(tmp23, tmp39, tmp26)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask & xmask, tmp43, _tmp42)
        tl.store(in_out_ptr0 + (r0_2 + 31040*x3), tmp40, r0_mask & xmask)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp42, xmask)
    tl.store(out_ptr2 + (x3), tmp42, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/2e/c2e6wcdalbrpn2idvupr4wdadcqafuyjnpyb6skom342oczamucc.py
# Topologically Sorted Source Nodes: [isfinite_4, all_2, ge, all_3], Original ATen: [aten.eq, aten.abs, aten.ne, aten.mul, aten.all, aten.ge]
# Source node to ATen node mapping:
#   all_2 => any_2, logical_not_2
#   all_3 => any_3, logical_not_4
#   ge => ge
#   isfinite_4 => abs_6, eq_6, mul_5, ne_4
# Graph fragment:
#   %eq_6 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%where_3, %where_3), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_3,), kwargs = {})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_6, inf), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_6, %ne_4), kwargs = {})
#   %logical_not_2 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_5,), kwargs = {})
#   %any_2 : [num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not_2,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%where_3, 0), kwargs = {})
#   %logical_not_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%ge,), kwargs = {})
#   %any_3 : [num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not_4,), kwargs = {})
triton_red_fused_abs_all_eq_ge_mul_ne_7 = async_compile.triton('triton_red_fused_abs_all_eq_ge_mul_ne_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_all_eq_ge_mul_ne_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1792, 'r0_': 58603776}}
)
@triton.jit
def triton_red_fused_abs_all_eq_ge_mul_ne_7(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 448
    r0_numel = 32703
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + 32703*x0
        tmp1 = tl.full([1, 1], 14650880, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + 32703*x0) % 14650880)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp3 == tmp3
        tmp5 = tl_math.abs(tmp3)
        tmp6 = float("inf")
        tmp7 = tmp5 != tmp6
        tmp8 = tmp4 & tmp7
        tmp9 = tmp8 == 0
        tmp10 = tl.full(tmp9.shape, False, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 | tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
        tmp15 = 0.0
        tmp16 = tmp3 >= tmp15
        tmp17 = tmp16 == 0
        tmp18 = tl.full(tmp17.shape, False, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 | tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp13 = triton_helpers.any(_tmp13.to(tl.int8), 1)[:, None].to(tl.int1)
    tmp21 = triton_helpers.any(_tmp21.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/b3/cb3uoanhp5x37azi5sp3jmhdhmzcvvmuo4taodwnyhywel45dxyh.py
# Topologically Sorted Source Nodes: [same_mask, all_2, and__1, all_3, and__2, gt_1, all_4, checks], Original ATen: [aten.all, aten.bitwise_and, aten.gt]
# Source node to ATen node mapping:
#   all_2 => logical_not_3
#   all_3 => logical_not_5
#   all_4 => any_4, logical_not_6, logical_not_7
#   and__1 => bitwise_and_1
#   and__2 => bitwise_and_2
#   checks => bitwise_and_3
#   gt_1 => gt_1
#   same_mask => logical_not_1
# Graph fragment:
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %logical_not_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_2,), kwargs = {})
#   %bitwise_and_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%logical_not_1, %logical_not_3), kwargs = {})
#   %logical_not_5 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_3,), kwargs = {})
#   %bitwise_and_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_1, %logical_not_5), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_3, 0), kwargs = {})
#   %logical_not_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%gt_1,), kwargs = {})
#   %any_4 : [num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not_6,), kwargs = {})
#   %logical_not_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_4,), kwargs = {})
#   %bitwise_and_3 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_2, %logical_not_7), kwargs = {})
triton_per_fused_all_bitwise_and_gt_8 = async_compile.triton('triton_per_fused_all_bitwise_and_gt_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i1', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(4,))]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_all_bitwise_and_gt_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 236}}
)
@triton.jit
def triton_per_fused_all_bitwise_and_gt_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 59
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (0)).to(tl.int1)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, 1])
    tmp11 = tl.load(in_ptr1 + (0)).to(tl.int1)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp15 = tl.load(in_ptr2 + (0)).to(tl.int1)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, 1])
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2 == 0
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, False)
    tmp7 = triton_helpers.any(tmp6, 1)[:, None]
    tmp10 = tmp9 == 0
    tmp13 = tmp12 == 0
    tmp14 = tmp10 & tmp13
    tmp17 = tmp16 == 0
    tmp18 = tmp14 & tmp17
    tmp19 = tmp7 == 0
    tmp20 = tmp18 & tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp20, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/65/c65l2wedncyt66z4spxwmh4cvalrbdc6bbiqxdq5qx53tshg65gw.py
# Topologically Sorted Source Nodes: [neg_2, seed, new_ones, scatter_add_], Original ATen: [aten.neg, aten.div, aten.new_ones, aten.scatter_add]
# Source node to ATen node mapping:
#   neg_2 => neg_2
#   new_ones => full_default_4
#   scatter_add_ => scatter_add
#   seed => div_1
# Graph fragment:
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_3,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg_2, %sum_4), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([59, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=2] = call_function[target=torch.ops.aten.scatter_add.default](args = (%div_1, -1, %unsqueeze, %full_default_4), kwargs = {})
triton_poi_fused_div_neg_new_ones_scatter_add_9 = async_compile.triton('triton_poi_fused_div_neg_new_ones_scatter_add_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_neg_new_ones_scatter_add_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 175810560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_neg_new_ones_scatter_add_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14650880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 248320
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp3 = (tmp1 / tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/cl/cclkskdx677squvf3lz5u3hfg3pxcbojlgtu42fp5b4y2f3wr67u.py
# Topologically Sorted Source Nodes: [neg_2, seed, new_ones, scatter_add_], Original ATen: [aten.neg, aten.div, aten.new_ones, aten.scatter_add]
# Source node to ATen node mapping:
#   neg_2 => neg_2
#   new_ones => full_default_4
#   scatter_add_ => scatter_add
#   seed => div_1
# Graph fragment:
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_3,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg_2, %sum_4), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([59, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=2] = call_function[target=torch.ops.aten.scatter_add.default](args = (%div_1, -1, %unsqueeze, %full_default_4), kwargs = {})
triton_poi_fused_div_neg_new_ones_scatter_add_10 = async_compile.triton('triton_poi_fused_div_neg_new_ones_scatter_add_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_neg_new_ones_scatter_add_10', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_neg_new_ones_scatter_add_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 59
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tl.atomic_add(out_ptr0 + (tmp0 + 248320*x0), tmp1, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/mn/cmnjmudl42mwno2v7ns4bebqbmz4yaokpkcx4ijzf7xomsap5q2y.py
# Topologically Sorted Source Nodes: [z0, z1, to, sub_1, mul_1, allocated], Original ATen: [aten._to_copy, aten.sub, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   allocated => sum_7
#   mul_1 => mul_6
#   sub_1 => sub_9
#   to => convert_element_type_2
#   z0 => convert_element_type
#   z1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.bfloat16), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, -1), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_1, -1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %convert_element_type), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_add, %sub_9), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_6, [-1]), kwargs = {})
triton_red_fused__to_copy_mul_sub_sum_11 = async_compile.triton('triton_red_fused__to_copy_mul_sub_sum_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sub_sum_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 117207040}}
)
@triton.jit
def triton_red_fused__to_copy_mul_sub_sum_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 472
    r0_numel = 31040
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    x2 = (xindex % 8)
    x3 = xindex // 8
    tmp4 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 31040*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 31040*x2 + 496640*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 31040*x2 + 496640*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 - tmp12
        tmp14 = tl_math.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
        tmp18 = tmp11 - tmp3
        tmp19 = tmp0 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
        tl.store(out_ptr0 + (r0_1 + 31040*x0), tmp1, r0_mask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/u7/cu7ybr44ids5h32bnlzbkefhm4ft6v7s7j4gej2uat2r74gkebjz.py
# Topologically Sorted Source Nodes: [z0, z1, log_softmax_2, gather, log_softmax_3, gather_1], Original ATen: [aten._to_copy, aten._log_softmax, aten.gather]
# Source node to ATen node mapping:
#   gather => gather
#   gather_1 => gather_1
#   log_softmax_2 => log_2, sub_6
#   log_softmax_3 => log_3, sub_8
#   z0 => convert_element_type
#   z1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, -1), kwargs = {})
#   %sub_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_2), kwargs = {})
#   %log_2 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_3,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_1, %log_2), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_6, -1, %unsqueeze_2), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_1, -1), kwargs = {})
#   %sub_tensor : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %getitem), kwargs = {})
#   %log_3 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log_3), kwargs = {})
#   %gather_1 : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_8, -1, %unsqueeze_3), kwargs = {})
triton_per_fused__log_softmax__to_copy_gather_12 = async_compile.triton('triton_per_fused__log_softmax__to_copy_gather_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy_gather_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy_gather_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 59
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_out_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 248320, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 496640*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp19 = tl_math.log(tmp9)
    tmp20 = tmp18 - tmp19
    tmp21 = tl.load(in_ptr4 + (tmp14 + 496640*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 - tmp23
    tmp25 = tl_math.log(tmp4)
    tmp26 = tmp24 - tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp26, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (59, 248320), (496640, 1))
    assert_size_stride(arg1_1, (59, 248320), (496640, 1))
    assert_size_stride(arg2_1, (59, ), (1, ))
    assert_size_stride(arg3_1, (248320, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((59, 1, 8), (8, 472, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_0.run(arg0_1, buf0, 472, 31040, stream=stream0)
        buf1 = empty_strided_cuda((59, 1), (1, 59), torch.float32)
        # Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_1.run(buf0, buf1, 59, 8, stream=stream0)
        buf4 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [z1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_0.run(arg1_1, buf4, 472, 31040, stream=stream0)
        buf5 = empty_strided_cuda((59, 1), (1, 59), torch.float32)
        # Topologically Sorted Source Nodes: [z1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_1.run(buf4, buf5, 59, 8, stream=stream0)
        buf2 = buf4; del buf4  # reuse
        buf29 = empty_strided_cuda((59, 1, 8), (8, 472, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_2.run(arg0_1, buf1, buf2, buf29, 472, 31040, stream=stream0)
        buf3 = empty_strided_cuda((59, 1), (1, 59), torch.float32)
        # Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_3.run(buf2, buf3, 59, 8, stream=stream0)
        buf30 = empty_strided_cuda((59, 1), (1, 59), torch.float32)
        # Topologically Sorted Source Nodes: [z0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_1.run(buf29, buf30, 59, 8, stream=stream0)
        buf6 = buf29; del buf29  # reuse
        buf34 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [z1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_2.run(arg1_1, buf5, buf6, buf34, 472, 31040, stream=stream0)
        buf7 = empty_strided_cuda((59, 1), (1, 59), torch.float32)
        # Topologically Sorted Source Nodes: [z1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_3.run(buf6, buf7, 59, 8, stream=stream0)
        buf35 = empty_strided_cuda((59, 1), (1, 59), torch.float32)
        # Topologically Sorted Source Nodes: [z1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_1.run(buf34, buf35, 59, 8, stream=stream0)
        buf18 = empty_strided_cuda((448, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [z0, lp0, z1, lp1, isfinite_2, isfinite_3, eq, same_mask], Original ATen: [aten._to_copy, aten._log_softmax, aten.eq, aten.abs, aten.ne, aten.mul, aten.all]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_abs_all_eq_mul_ne_4.run(arg0_1, buf1, buf3, arg1_1, buf5, buf7, buf18, 448, 32703, stream=stream0)
        buf19 = empty_strided_cuda((), (), torch.bool)
        # Topologically Sorted Source Nodes: [z0, lp0, z1, lp1, isfinite_2, isfinite_3, eq, same_mask], Original ATen: [aten._to_copy, aten._log_softmax, aten.eq, aten.abs, aten.ne, aten.mul, aten.all]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_abs_all_eq_mul_ne_5.run(buf18, buf19, 1, 448, stream=stream0)
        buf9 = empty_strided_cuda((59, 248320), (248320, 1), torch.float32)
        buf10 = buf9; del buf9  # reuse
        buf11 = buf34; del buf34  # reuse
        buf26 = reinterpret_tensor(buf6, (59, 8), (8, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [z0, lp0, z1, lp1, maximum, exp, isfinite, isfinite_1, valid, sub, abs_1, zeros_like, distance, eq_1, ones_like_1, neg, expm1, neg_1, gt, ones_like, denominator, ratio, ratio_1, mul, zeros_like_1, mean, sum_2, sum_1], Original ATen: [aten._to_copy, aten._log_softmax, aten.maximum, aten.exp, aten.eq, aten.abs, aten.ne, aten.mul, aten.bitwise_and, aten.sub, aten.zeros_like, aten.where, aten.ones_like, aten.neg, aten.expm1, aten.gt, aten.div, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_abs_bitwise_and_div_eq_exp_expm1_gt_maximum_mul_ne_neg_ones_like_sub_sum_where_zeros_like_6.run(buf10, arg0_1, buf1, buf3, arg1_1, buf5, buf7, buf11, buf26, 472, 31040, stream=stream0)
        del buf1
        del buf3
        buf12 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_3.run(buf11, buf12, 59, 8, stream=stream0)
        buf27 = reinterpret_tensor(buf5, (59, ), (1, ), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_3.run(buf26, buf27, 59, 8, stream=stream0)
        buf21 = buf18; del buf18  # reuse
        buf24 = empty_strided_cuda((448, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [isfinite_4, all_2, ge, all_3], Original ATen: [aten.eq, aten.abs, aten.ne, aten.mul, aten.all, aten.ge]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_all_eq_ge_mul_ne_7.run(buf10, buf21, buf24, 448, 32703, stream=stream0)
        buf22 = empty_strided_cuda((), (), torch.bool)
        # Topologically Sorted Source Nodes: [isfinite_4, all_2], Original ATen: [aten.eq, aten.abs, aten.ne, aten.mul, aten.all]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_abs_all_eq_mul_ne_5.run(buf21, buf22, 1, 448, stream=stream0)
        del buf21
        buf25 = empty_strided_cuda((), (), torch.bool)
        # Topologically Sorted Source Nodes: [ge, all_3], Original ATen: [aten.ge, aten.all]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_abs_all_eq_mul_ne_5.run(buf24, buf25, 1, 448, stream=stream0)
        del buf24
        buf41 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [same_mask, all_2, and__1, all_3, and__2, gt_1, all_4, checks], Original ATen: [aten.all, aten.bitwise_and, aten.gt]
        stream0 = get_raw_stream(0)
        triton_per_fused_all_bitwise_and_gt_8.run(buf41, buf27, buf22, buf25, 1, 59, stream=stream0)
        del buf22
        del buf25
        del buf27
        buf13 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [neg_2, seed, new_ones, scatter_add_], Original ATen: [aten.neg, aten.div, aten.new_ones, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_neg_new_ones_scatter_add_9.run(buf13, buf12, 14650880, stream=stream0)
        # Topologically Sorted Source Nodes: [neg_2, seed, new_ones, scatter_add_], Original ATen: [aten.neg, aten.div, aten.new_ones, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_neg_new_ones_scatter_add_10.run(arg2_1, buf13, 59, stream=stream0)
        buf15 = empty_strided_cuda((1, 59, 248320), (14650880, 248320, 1), torch.bfloat16)
        buf31 = reinterpret_tensor(buf26, (59, 1, 8), (8, 472, 1), 0); del buf26  # reuse
        buf36 = buf11; del buf11  # reuse
        buf39 = empty_strided_cuda((59, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z0, z1, to, sub_1, mul_1, allocated], Original ATen: [aten._to_copy, aten.sub, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_mul_sub_sum_11.run(buf13, arg0_1, buf30, arg1_1, buf35, buf15, buf31, buf36, buf39, 472, 31040, stream=stream0)
        del buf13
        buf16 = empty_strided_cuda((1, 59, 4096), (241664, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to, bmm], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf15, reinterpret_tensor(arg3_1, (1, 248320, 4096), (1017118720, 4096, 1), 0), out_dtype=torch.float32, out=buf16)
        del arg3_1
        del buf15
        buf33 = reinterpret_tensor(buf30, (59, 1), (1, 1), 0); del buf30  # reuse
        buf38 = reinterpret_tensor(buf35, (59, 1), (1, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [z0, z1, log_softmax_2, gather, log_softmax_3, gather_1], Original ATen: [aten._to_copy, aten._log_softmax, aten.gather]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_gather_12.run(buf33, buf38, buf36, buf31, arg2_1, arg0_1, arg1_1, 59, 8, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del buf31
        del buf36
        buf40 = reinterpret_tensor(buf12, (59, ), (1, ), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [z0, z1, sub_1, mul_1, allocated], Original ATen: [aten._to_copy, aten.sub, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_3.run(buf39, buf40, 59, 8, stream=stream0)
        del buf39
    return (reinterpret_tensor(buf16, (59, 4096), (4096, 1), 0), buf41, reinterpret_tensor(buf33, (59, ), (1, ), 0), reinterpret_tensor(buf38, (59, ), (1, ), 0), buf40, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((59, 248320), (496640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((59, 248320), (496640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((59, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((248320, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
