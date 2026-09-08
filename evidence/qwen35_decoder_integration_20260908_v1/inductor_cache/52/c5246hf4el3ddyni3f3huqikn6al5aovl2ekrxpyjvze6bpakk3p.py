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


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/7a/c7awenfafvbiiygcsz4tv4gkyhvybisame3t7yrrd4tiumprjuve.py
# Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 39649280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4956160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/nv/cnvsh5bjoxpqv6x2r5fxnvq6dfzllkc2q25s3fuuc7vjx3aidk5c.py
# Topologically Sorted Source Nodes: [float_1, float_2, to_2, up0, up1, add, mul, mg, gate1, gate0, difference, ne, sub_2, ne_1, ones_like, where, truediv, sigmoid, sub_1, mul_4, add_2, derivative, multiplier, m_gate, to_4], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sub, aten.ne, aten.ones_like, aten.where, aten.div, aten.sigmoid, aten.rsub]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   derivative => mul_5
#   difference => sub
#   float_1 => convert_element_type_3
#   float_2 => convert_element_type_4
#   gate0 => convert_element_type_5
#   gate1 => convert_element_type_6
#   m_gate => mul_6
#   mg => mul_1
#   mul => mul
#   mul_4 => mul_4
#   multiplier => where_1
#   ne => ne
#   ne_1 => ne_1
#   ones_like => full_default
#   sigmoid => sigmoid
#   sub_1 => sub_1
#   sub_2 => sub_2
#   to_2 => convert_element_type_9
#   to_4 => convert_element_type_12
#   truediv => div
#   up0 => convert_element_type_7
#   up1 => convert_element_type_8
#   where => where
# Graph fragment:
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.bfloat16), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, %convert_element_type_8), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %add), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 0.5), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %convert_element_type_5 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %convert_element_type_5), kwargs = {})
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%sub, 0), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %convert_element_type_3), kwargs = {})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%sub, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 605, 12288], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %sub, %full_default), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %where), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_5,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %sub_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %add_2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %div, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %where_1), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 356843520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14868480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    x2 = (xindex % 7434240)
    x3 = xindex // 7434240
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 12288*((x1 % 605)) + 14868480*(x1 // 605)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 12288*((x1 % 605)) + 14868480*(x1 // 605)), None).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp19 = tl.load(in_ptr6 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp15 = tmp0 * tmp14
    tmp16 = tmp15 * tmp7
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 - tmp20
    tmp22 = 0.0
    tmp23 = tmp21 != tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 - tmp27
    tmp29 = 1.0
    tmp30 = tl.where(tmp23, tmp21, tmp29)
    tmp31 = (tmp28 / tmp30)
    tmp32 = tl.sigmoid(tmp20)
    tmp33 = tmp29 - tmp32
    tmp34 = tmp20 * tmp33
    tmp35 = tmp34 + tmp29
    tmp36 = tmp32 * tmp35
    tmp37 = tl.where(tmp23, tmp31, tmp36)
    tmp38 = tmp16 * tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp9, None)
    tl.store(out_ptr2 + (x4), tmp39, None)
''', device_str='cuda')


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/mr/cmrofxoct2y24nvzhdoxy73iynbamiblalrms2fb4xdchqnogbmq.py
# Topologically Sorted Source Nodes: [add_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_3 => add_3
# Graph fragment:
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %view_5), kwargs = {})
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 79298560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4956160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 605, 4096), (2478080, 4096, 1))
    assert_size_stride(arg1_1, (4096, 12288), (12288, 1))
    assert_size_stride(arg2_1, (2, 605, 12288), (14868480, 12288, 1))
    assert_size_stride(arg3_1, (2, 605, 12288), (14868480, 12288, 1))
    assert_size_stride(arg4_1, (2, 605, 12288), (14868480, 12288, 1))
    assert_size_stride(arg5_1, (2, 605, 12288), (14868480, 12288, 1))
    assert_size_stride(arg6_1, (2, 605, 12288), (14868480, 12288, 1))
    assert_size_stride(arg7_1, (2, 605, 12288), (14868480, 12288, 1))
    assert_size_stride(arg8_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg9_1, (12288, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1210, 4096), (4956160, 4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, 4956160, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((1, 1210, 12288), (14868480, 12288, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to, bmm], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf0, reinterpret_tensor(arg1_1, (1, 4096, 12288), (50331648, 12288, 1), 0), out_dtype=torch.float32, out=buf1)
        del arg1_1
        del buf0
        buf2 = empty_strided_cuda((1, 1210, 12288), (14868480, 12288, 1), torch.bfloat16)
        buf5 = empty_strided_cuda((1, 1210, 12288), (14868480, 12288, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_1, float_2, to_2, up0, up1, add, mul, mg, gate1, gate0, difference, ne, sub_2, ne_1, ones_like, where, truediv, sigmoid, sub_1, mul_4, add_2, derivative, multiplier, m_gate, to_4], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sub, aten.ne, aten.ones_like, aten.where, aten.div, aten.sigmoid, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1.run(buf1, arg2_1, arg3_1, arg6_1, arg7_1, arg5_1, arg4_1, buf2, buf5, 14868480, stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf1
        buf3 = empty_strided_cuda((1, 1210, 4096), (4956160, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to_2, bmm_1], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf2, reinterpret_tensor(arg8_1, (1, 12288, 4096), (50331648, 4096, 1), 0), out_dtype=torch.float32, out=buf3)
        del arg8_1
        del buf2
        buf6 = empty_strided_cuda((1, 1210, 4096), (4956160, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to_4, bmm_2], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf5, reinterpret_tensor(arg9_1, (1, 12288, 4096), (50331648, 4096, 1), 0), out_dtype=torch.float32, out=buf6)
        del arg9_1
        del buf5
        buf7 = reinterpret_tensor(buf3, (2, 605, 4096), (2478080, 4096, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [add_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf7, buf6, 4956160, stream=stream0)
        del buf6
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 605, 4096), (2478080, 4096, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4096, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2, 605, 12288), (14868480, 12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2, 605, 12288), (14868480, 12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2, 605, 12288), (14868480, 12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((2, 605, 12288), (14868480, 12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((2, 605, 12288), (14868480, 12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((2, 605, 12288), (14868480, 12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
