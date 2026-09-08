# AOT ID: ['2_inference']
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
#   to => convert_element_type_4
# Graph fragment:
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.bfloat16), kwargs = {})
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


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/f3/cf32a64prmbh22aylnovf7n4suf6lrakca7srsvdvvt7octemst3.py
# Topologically Sorted Source Nodes: [sigmoid_1, s1, contiguous, float_5, mul_1, sigmoid, s0, sub_2, g1f, g0f, delta, nonzero, ones_like, where, truediv, derivative, sub, mul_2, where_1, mgate], Original ATen: [aten.sigmoid, aten._to_copy, aten.clone, aten.mul, aten.sub, aten.ne, aten.ones_like, aten.where, aten.div, aten.rsub]
# Source node to ATen node mapping:
#   contiguous => clone
#   delta => sub_1
#   derivative => sigmoid_2
#   float_5 => convert_element_type_7
#   g0f => convert_element_type_2
#   g1f => convert_element_type_3
#   mgate => mul_3
#   mul_1 => mul_1
#   mul_2 => mul_2
#   nonzero => ne
#   ones_like => full_default
#   s0 => convert_element_type
#   s1 => convert_element_type_1
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   sub => sub
#   sub_2 => sub_2
#   truediv => div
#   where => where
#   where_1 => where_1
# Graph fragment:
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%slice_2,), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_1, torch.float32), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %convert_element_type_7), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%slice_1,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %convert_element_type), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_2, torch.float32), kwargs = {})
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, %convert_element_type_2), kwargs = {})
#   %ne : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%sub_1, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 605, 16, 256], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %sub_1, %full_default), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %where), kwargs = {})
#   %sigmoid_2 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_2, %sub), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %div, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %where_1), kwargs = {})
triton_poi_fused__to_copy_clone_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1 = async_compile.triton('triton_poi_fused__to_copy_clone_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128860160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4956160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x0 = (xindex % 256)
    x3 = xindex // 2478080
    x4 = ((xindex // 256) % 9680)
    x1 = ((xindex // 256) % 16)
    x2 = ((xindex // 4096) % 605)
    x6 = (xindex % 2478080)
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + 512*x4 + 9912320*x3), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x6 + 4956160*x3), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (256 + x0 + 512*x4 + 9912320*x3), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 * tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp1.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = 0.0
    tmp13 = tmp11 != tmp12
    tmp14 = tl.sigmoid(tmp9)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp3 - tmp15
    tmp17 = 1.0
    tmp18 = tl.where(tmp13, tmp11, tmp17)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.sigmoid(tmp10)
    tmp21 = tmp17 - tmp20
    tmp22 = tmp20 * tmp21
    tmp23 = tl.where(tmp13, tmp19, tmp22)
    tmp24 = tmp7 * tmp23
    tl.store(out_ptr0 + (x0 + 256*x2 + 154880*x1 + 2478080*x3), tmp4, None)
    tl.store(out_ptr1 + (x5), tmp24, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 605, 4096), (2478080, 4096, 1))
    assert_size_stride(arg1_1, (2, 605, 8192), (9912320, 8192, 1))
    assert_size_stride(arg2_1, (2, 605, 8192), (9912320, 8192, 1))
    assert_size_stride(arg3_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg4_1, (2, 605, 16, 256), (4956160, 4096, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1210, 4096), (4956160, 4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, 4956160, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((1, 1210, 4096), (4956160, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to, bmm], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm_dtype(buf0, reinterpret_tensor(arg3_1, (1, 4096, 4096), (16777216, 4096, 1), 0), out_dtype=torch.float32, out=buf1)
        del arg3_1
        del buf0
        buf2 = empty_strided_cuda((2, 16, 605, 256), (2478080, 154880, 256, 1), torch.float32)
        buf3 = empty_strided_cuda((2, 605, 16, 256), (2478080, 4096, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_1, s1, contiguous, float_5, mul_1, sigmoid, s0, sub_2, g1f, g0f, delta, nonzero, ones_like, where, truediv, derivative, sub, mul_2, where_1, mgate], Original ATen: [aten.sigmoid, aten._to_copy, aten.clone, aten.mul, aten.sub, aten.ne, aten.ones_like, aten.where, aten.div, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1.run(buf1, arg2_1, arg4_1, arg1_1, buf2, buf3, 4956160, stream=stream0)
        del arg1_1
        del arg2_1
        del arg4_1
        del buf1
    return (buf2, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 605, 4096), (2478080, 4096, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 605, 8192), (9912320, 8192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2, 605, 8192), (9912320, 8192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2, 605, 16, 256), (4956160, 4096, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
