# AOT ID: ['3_inference']
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


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/nk/cnkorodutz2m26mot5g42ff6k23m5zmq6tyx2tfkv34tdtzdo6jt.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub, %slice_2], -1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 60712960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4956160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x4 = xindex // 256
    x1 = ((xindex // 256) % 605)
    x3 = xindex // 2478080
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (64*x1 + 77440*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = x0
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 32, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp14 & tmp4
    tmp16 = tl.load(in_ptr0 + (32 + 256*x4 + (x0)), tmp15, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (32 + 64*x1 + 77440*x3 + (x0)), tmp15, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = -tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp15, tmp21, tmp22)
    tmp24 = tmp10 >= tmp13
    tmp25 = tl.full([1], 64, tl.int64)
    tmp26 = tmp10 < tmp25
    tmp27 = tmp24 & tmp4
    tmp28 = tl.load(in_ptr0 + (256*x4 + ((-32) + (x0))), tmp27, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (64*x1 + 77440*x3 + ((-32) + (x0))), tmp27, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp27, tmp32, tmp33)
    tmp35 = tl.where(tmp14, tmp23, tmp34)
    tmp36 = tmp9 - tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp4, tmp36, tmp37)
    tmp39 = tmp0 >= tmp3
    tmp40 = tl.full([1], 256, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr0 + (64 + 256*x4 + ((-64) + x0)), tmp39, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp39, tmp43, tmp44)
    tmp46 = tl.where(tmp4, tmp38, tmp45)
    tl.store(out_ptr0 + (x5), tmp46, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/c2/cc2qmuth6os24toypqcugyjwm76shm5uuvvvvw4to5wb4xlkperc.py
# Topologically Sorted Source Nodes: [float_8, square, mean, add_1, r0, truediv, float_9, square_1, mean_1, add_2, r1, truediv_1, add_3, inverse_mean, float_10, add, weighted, mul_10, add_5, xbar, mul_5, add_4, mul_6, inverse_secant, mul_9, coefficient, mul_11, mul_12, sum_3, mul_13, mq_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.sqrt, aten.reciprocal, aten.mul, aten.div, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   coefficient => div
#   float_10 => convert_element_type_9
#   float_8 => convert_element_type_7
#   float_9 => convert_element_type_8
#   inverse_mean => mul_6
#   inverse_secant => mul_9, reciprocal_2
#   mean => mean
#   mean_1 => mean_1
#   mq_2 => add_6
#   mul_10 => mul_13
#   mul_11 => mul_14
#   mul_12 => mul_15
#   mul_13 => mul_16
#   mul_5 => mul_7
#   mul_6 => mul_8
#   mul_9 => mul_12
#   r0 => sqrt
#   r1 => sqrt_1
#   square => pow_1
#   square_1 => pow_2
#   sum_3 => sum_3
#   truediv => mul_4, reciprocal
#   truediv_1 => mul_5, reciprocal_1
#   weighted => mul_11
#   xbar => mul_10
# Graph fragment:
#   %convert_element_type_7 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_7, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %sqrt : [num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1,), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %sqrt_1 : [num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.5), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1), kwargs = {})
#   %mul_11 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_1, %add), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %mul_11), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, %convert_element_type_8), kwargs = {})
#   %mul_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt, %sqrt_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, %sqrt_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_4), kwargs = {})
#   %reciprocal_2 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%mul_8,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, -1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 2), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_12, 256), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %div), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [-1], True), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %sum_3), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_16), kwargs = {})
triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_1 = async_compile.triton('triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 3, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 79299072}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 19360
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 9680)
    x1 = xindex // 9680
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = (xindex % 16)
    x4 = ((xindex // 16) % 605)
    _tmp23 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 256*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_2 + 256*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r0_2 + 256*x4 + 154880*x3 + 2478080*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp12 = tmp1 + tmp7
        tmp13 = 0.5
        tmp14 = tmp12 * tmp13
        tmp17 = tmp16.to(tl.float32)
        tmp18 = 1.0
        tmp19 = tmp17 + tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tmp14 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(r0_mask & xmask, tmp24, _tmp23)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp42 = tl.load(in_ptr2 + (r0_2 + 256*x4 + 154880*x3 + 2478080*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tl.load(in_ptr0 + (r0_2 + 256*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr1 + (r0_2 + 256*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = 256.0
        tmp26 = (tmp4 / tmp25)
        tmp27 = 1e-06
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.sqrt(tmp28)
        tmp30 = tl.full([1, 1], 1, tl.int32)
        tmp31 = (tmp30 / tmp29)
        tmp32 = 1.0
        tmp33 = tmp31 * tmp32
        tmp34 = (tmp10 / tmp25)
        tmp35 = tmp34 + tmp27
        tmp36 = libdevice.sqrt(tmp35)
        tmp37 = (tmp30 / tmp36)
        tmp38 = tmp37 * tmp32
        tmp39 = tmp33 + tmp38
        tmp40 = 0.5
        tmp41 = tmp39 * tmp40
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp44 + tmp32
        tmp46 = tmp42 * tmp45
        tmp47 = tmp41 * tmp46
        tmp49 = tmp48.to(tl.float32)
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp49 + tmp51
        tmp53 = tmp52 * tmp40
        tmp54 = tmp29 * tmp36
        tmp55 = tmp29 + tmp36
        tmp56 = tmp54 * tmp55
        tmp57 = (tmp30 / tmp56)
        tmp58 = -1.0
        tmp59 = tmp57 * tmp58
        tmp60 = 2.0
        tmp61 = tmp59 * tmp60
        tmp62 = 0.00390625
        tmp63 = tmp61 * tmp62
        tmp64 = tmp53 * tmp63
        tmp65 = tmp64 * tmp23
        tmp66 = tmp47 + tmp65
        tl.store(out_ptr3 + (r0_2 + 256*x5), tmp66, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/6n/c6n4ltpcpcokhkt75befzokj6jvyobi3dnrgx6tzqiyr5muh7w7c.py
# Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to => convert_element_type_13
# Graph fragment:
#   %convert_element_type_13 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 118947840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9912320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    tmp0 = (x2 % 512)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*(x0 // 512) + 4096*x1 + ((x0 % 512))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (256*(x0 // 512) + 4096*x1 + ((-256) + ((x0 % 512)))), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/xm/cxm6pzn5cncxpxvrargkwba3opds4lrau6pzkfbpceshybsc4j3g.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %getitem_2], -1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7744000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 309760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 605)
    x4 = xindex // 38720
    x3 = xindex // 154880
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32 + 256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (154912 + 256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.load(in_ptr0 + (309792 + 256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.load(in_ptr0 + (464672 + 256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tl.load(in_ptr1 + (32 + 64*x1 + 77440*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 * tmp17
    tmp19 = -tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = tl.full([1], 64, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (256*x1 + 619520*x4 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr0 + (154880 + 256*x1 + 619520*x4 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 + tmp28
    tmp30 = tl.load(in_ptr0 + (309760 + 256*x1 + 619520*x4 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tl.load(in_ptr0 + (464640 + 256*x1 + 619520*x4 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 + tmp34
    tmp36 = tl.load(in_ptr1 + (64*x1 + 77440*x3 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp22, tmp38, tmp39)
    tmp41 = tl.where(tmp4, tmp21, tmp40)
    tl.store(out_ptr0 + (x5), tmp41, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/ib/cibry6sjdakocnn6s5lzjerwzstrxytttlru7xfkyhcrobty4hbx.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_1, %slice_4], -1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25400320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1239040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 605)
    x4 = xindex // 154880
    x3 = xindex // 619520
    x5 = xindex // 256
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (154880 + 256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.load(in_ptr0 + (309760 + 256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.load(in_ptr0 + (464640 + 256*x1 + 619520*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tl.load(in_ptr1 + (64*x1 + 77440*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 * tmp17
    tmp19 = tl.load(in_ptr2 + (64*x5 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 256, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr0 + (64 + 256*x1 + 619520*x4 + ((-64) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.load(in_ptr0 + (154944 + 256*x1 + 619520*x4 + ((-64) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 + tmp29
    tmp31 = tl.load(in_ptr0 + (309824 + 256*x1 + 619520*x4 + ((-64) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 + tmp32
    tmp34 = tl.load(in_ptr0 + (464704 + 256*x1 + 619520*x4 + ((-64) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 + tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp23, tmp36, tmp37)
    tmp39 = tl.where(tmp4, tmp22, tmp38)
    tl.store(out_ptr0 + (x6), tmp39, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/zf/czf7b3toqkfmpyozpjmsi4q644zpufyex35eaunit4i6ef64zloz.py
# Topologically Sorted Source Nodes: [float_11, square_2, mean_2, add_8, r0_1, truediv_4, float_12, square_3, mean_3, add_9, r1_1, truediv_5, add_10, inverse_mean_1, float_13, add_7, weighted_1, mul_20, add_12, xbar_1, mul_15, add_11, mul_16, inverse_secant_1, mul_19, coefficient_1, mul_21, mul_22, sum_4, mul_23, mk_2, to_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.sqrt, aten.reciprocal, aten.mul, aten.div, aten.sum]
# Source node to ATen node mapping:
#   add_10 => add_10
#   add_11 => add_11
#   add_12 => add_12
#   add_7 => add_7
#   add_8 => add_8
#   add_9 => add_9
#   coefficient_1 => div_1
#   float_11 => convert_element_type_10
#   float_12 => convert_element_type_11
#   float_13 => convert_element_type_12
#   inverse_mean_1 => mul_19
#   inverse_secant_1 => mul_22, reciprocal_5
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mk_2 => add_13
#   mul_15 => mul_20
#   mul_16 => mul_21
#   mul_19 => mul_25
#   mul_20 => mul_26
#   mul_21 => mul_27
#   mul_22 => mul_28
#   mul_23 => mul_29
#   r0_1 => sqrt_2
#   r1_1 => sqrt_3
#   square_2 => pow_3
#   square_3 => pow_4
#   sum_4 => sum_4
#   to_2 => convert_element_type_16
#   truediv_4 => mul_17, reciprocal_3
#   truediv_5 => mul_18, reciprocal_4
#   weighted_1 => mul_24
#   xbar_1 => mul_23
# Graph fragment:
#   %convert_element_type_10 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.float32), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_10, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %sqrt_2 : [num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_3 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %convert_element_type_11 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_11, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %sqrt_3 : [num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_9,), kwargs = {})
#   %reciprocal_4 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %mul_18), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.float32), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_12, 1), kwargs = {})
#   %mul_24 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %add_7), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %mul_24), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_10, %convert_element_type_11), kwargs = {})
#   %mul_23 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 0.5), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_2, %sqrt_3), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_2, %sqrt_3), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %add_11), kwargs = {})
#   %reciprocal_5 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%mul_21,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, -1), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, 2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_25, 256), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %div_1), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %mul_24), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_28, [-1], True), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %sum_4), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_29), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_7, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_5 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 19825152}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr4, xnumel, r0_numel):
    xnumel = 4840
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 2420)
    x1 = xindex // 2420
    x3 = (xindex % 4)
    x4 = ((xindex // 4) % 605)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 256*x0 + 1239040*x1), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (r0_2 + 256*x0 + 1239040*x1), None).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r0_2 + 256*x4 + 154880*x3 + 619520*x1), None)
    tmp16 = tl.load(in_ptr3 + (r0_2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp1 + tmp7
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp15 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 256.0
    tmp26 = (tmp5 / tmp25)
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1], 1, tl.int32)
    tmp31 = (tmp30 / tmp29)
    tmp32 = tmp31 * tmp18
    tmp33 = (tmp11 / tmp25)
    tmp34 = tmp33 + tmp27
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = (tmp30 / tmp35)
    tmp37 = tmp36 * tmp18
    tmp38 = tmp32 + tmp37
    tmp39 = tmp38 * tmp13
    tmp40 = tmp39 * tmp20
    tmp41 = tmp29 * tmp35
    tmp42 = tmp29 + tmp35
    tmp43 = tmp41 * tmp42
    tmp44 = (tmp30 / tmp43)
    tmp45 = -1.0
    tmp46 = tmp44 * tmp45
    tmp47 = 2.0
    tmp48 = tmp46 * tmp47
    tmp49 = 0.00390625
    tmp50 = tmp48 * tmp49
    tmp51 = tmp14 * tmp50
    tmp52 = tmp51 * tmp24
    tmp53 = tmp40 + tmp52
    tmp54 = tmp53.to(tl.float32)
    tl.store(out_ptr4 + (r0_2 + 256*x5), tmp54, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/j2/cj2jrh3kds3d53bvpjircc7rbgytoby6ys67a347ns5h3ckmxu7o.py
# Topologically Sorted Source Nodes: [to_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_4 => convert_element_type_19
# Graph fragment:
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14868480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1239040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256*((x1 % 605)) + 619520*(x0 // 256) + 2478080*(x1 // 605) + ((x0 % 256))), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (154880 + 256*((x1 % 605)) + 619520*(x0 // 256) + 2478080*(x1 // 605) + ((x0 % 256))), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (309760 + 256*((x1 % 605)) + 619520*(x0 // 256) + 2478080*(x1 // 605) + ((x0 % 256))), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (464640 + 256*((x1 % 605)) + 619520*(x0 // 256) + 2478080*(x1 // 605) + ((x0 % 256))), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/gb/cgbqmfbv4qrtpe436pwc3hxuo3dolwfvz3ytynxfm2kws7fz2qxp.py
# Topologically Sorted Source Nodes: [add_14, add_15], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_14 => add_14
#   add_15 => add_15
# Graph fragment:
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_8), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %view_10), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 99123200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4956160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 16, 605, 256), (2478080, 154880, 256, 1))
    assert_size_stride(arg1_1, (2, 16, 605, 256), (2478080, 154880, 256, 1))
    assert_size_stride(arg2_1, (2, 16, 605, 256), (2478080, 154880, 256, 1))
    assert_size_stride(arg3_1, (2, 605, 64), (77440, 64, 1))
    assert_size_stride(arg4_1, (2, 605, 64), (77440, 64, 1))
    assert_size_stride(arg5_1, (2, 605, 16, 256), (4956160, 4096, 256, 1))
    assert_size_stride(arg6_1, (2, 605, 16, 256), (4956160, 4096, 256, 1))
    assert_size_stride(arg7_1, (256, ), (1, ))
    assert_size_stride(arg8_1, (2, 605, 4, 256), (1239040, 1024, 256, 1))
    assert_size_stride(arg9_1, (2, 605, 4, 256), (1239040, 1024, 256, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (2, 605, 16, 256), (2478080, 4096, 256, 1))
    assert_size_stride(arg12_1, (8192, 4096), (4096, 1))
    assert_size_stride(arg13_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg14_1, (1024, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((2, 16, 605, 256), (2478080, 154880, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg0_1, arg3_1, arg4_1, buf2, 4956160, stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((2, 605, 16, 256), (2478080, 4096, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_8, square, mean, add_1, r0, truediv, float_9, square_1, mean_1, add_2, r1, truediv_1, add_3, inverse_mean, float_10, add, weighted, mul_10, add_5, xbar, mul_5, add_4, mul_6, inverse_secant, mul_9, coefficient, mul_11, mul_12, sum_3, mul_13, mq_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.sqrt, aten.reciprocal, aten.mul, aten.div, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_1.run(arg5_1, arg6_1, buf2, arg7_1, buf4, 19360, 256, stream=stream0)
        del arg5_1
        del arg6_1
        del arg7_1
        buf5 = empty_strided_cuda((1, 1210, 8192), (9912320, 8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf4, arg11_1, buf5, 9912320, stream=stream0)
        del arg11_1
        buf6 = reinterpret_tensor(buf4, (1, 1210, 4096), (4956160, 4096, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [to, bmm], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf5, reinterpret_tensor(arg12_1, (1, 8192, 4096), (33554432, 4096, 1), 0), out_dtype=torch.float32, out=buf6)
        del arg12_1
        del buf5
        buf9 = empty_strided_cuda((2, 4, 605, 64), (154880, 38720, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(arg1_1, arg4_1, buf9, 309760, stream=stream0)
        del arg4_1
        buf10 = empty_strided_cuda((2, 4, 605, 256), (619520, 154880, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(arg1_1, arg3_1, buf9, buf10, 1239040, stream=stream0)
        del arg1_1
        del arg3_1
        del buf9
        buf13 = empty_strided_cuda((1, 1210, 1024), (1239040, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_11, square_2, mean_2, add_8, r0_1, truediv_4, float_12, square_3, mean_3, add_9, r1_1, truediv_5, add_10, inverse_mean_1, float_13, add_7, weighted_1, mul_20, add_12, xbar_1, mul_15, add_11, mul_16, inverse_secant_1, mul_19, coefficient_1, mul_21, mul_22, sum_4, mul_23, mk_2, to_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.sqrt, aten.reciprocal, aten.mul, aten.div, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_5.run(arg8_1, arg9_1, buf10, arg10_1, buf13, 4840, 256, stream=stream0)
        del arg10_1
        del arg8_1
        del arg9_1
        del buf10
        buf14 = reinterpret_tensor(buf2, (1, 1210, 4096), (4956160, 4096, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [to_2, bmm_1], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf13, reinterpret_tensor(arg13_1, (1, 1024, 4096), (4194304, 4096, 1), 0), out_dtype=torch.float32, out=buf14)
        del arg13_1
        buf15 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [to_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg2_1, buf15, 1239040, stream=stream0)
        del arg2_1
        buf16 = empty_strided_cuda((1, 1210, 4096), (4956160, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to_4, bmm_2], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf15, reinterpret_tensor(arg14_1, (1, 1024, 4096), (4194304, 4096, 1), 0), out_dtype=torch.float32, out=buf16)
        del arg14_1
        del buf15
        buf17 = reinterpret_tensor(buf6, (2, 605, 4096), (2478080, 4096, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [add_14, add_15], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_7.run(buf17, buf14, buf16, 4956160, stream=stream0)
        del buf14
        del buf16
    return (buf17, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 16, 605, 256), (2478080, 154880, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2, 16, 605, 256), (2478080, 154880, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2, 16, 605, 256), (2478080, 154880, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2, 605, 64), (77440, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2, 605, 64), (77440, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((2, 605, 16, 256), (4956160, 4096, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((2, 605, 16, 256), (4956160, 4096, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((2, 605, 4, 256), (1239040, 1024, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((2, 605, 4, 256), (1239040, 1024, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((2, 605, 16, 256), (2478080, 4096, 256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
