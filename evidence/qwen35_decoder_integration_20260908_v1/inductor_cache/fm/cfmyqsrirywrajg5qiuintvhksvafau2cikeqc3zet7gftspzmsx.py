# AOT ID: ['1_inference']
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


# kernel path: ${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache/kc/ckcucvdevsanwtvbhad2nmrvv7m6gn4ds2gzlmyjemrgrqhustqy.py
# Topologically Sorted Source Nodes: [float_1, square, mean, add_1, r0, truediv, float_2, square_1, mean_1, add_2, r1, truediv_1, add_3, inverse_mean, float_3, add, weighted, mul_6, add_5, xbar, mul_1, add_4, mul_2, inverse_secant, mul_5, coefficient, mul_7, mul_8, sum_1, mul_9, add_6, add_7], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.sqrt, aten.reciprocal, aten.mul, aten.div, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   add_7 => add_7
#   coefficient => div
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   float_3 => convert_element_type_2
#   inverse_mean => mul_2
#   inverse_secant => mul_5, reciprocal_2
#   mean => mean
#   mean_1 => mean_1
#   mul_1 => mul_3
#   mul_2 => mul_4
#   mul_5 => mul_8
#   mul_6 => mul_9
#   mul_7 => mul_10
#   mul_8 => mul_11
#   mul_9 => mul_12
#   r0 => sqrt
#   r1 => sqrt_1
#   square => pow_1
#   square_1 => pow_2
#   sum_1 => sum_1
#   truediv => mul, reciprocal
#   truediv_1 => mul_1, reciprocal_1
#   weighted => mul_7
#   xbar => mul_6
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %sqrt : [num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1,), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %sqrt_1 : [num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.5), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %add), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %mul_7), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.5), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt, %sqrt_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, %sqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %add_4), kwargs = {})
#   %reciprocal_2 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, -1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 2), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_8, 4096), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %div), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_11, [-1], True), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %sum_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_12), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %add_6), kwargs = {})
triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_0 = async_compile.triton('triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 3, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 99131392}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1210
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 605)
    x1 = xindex // 605
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp23 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4096*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_2 + 4096*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r0_2 + 4096*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp42 = tl.load(in_ptr2 + (r0_2 + 4096*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tl.load(in_ptr0 + (r0_2 + 4096*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr1 + (r0_2 + 4096*x0 + 4956160*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr4 + (r0_2 + 4096*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = 4096.0
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
        tmp62 = 0.000244140625
        tmp63 = tmp61 * tmp62
        tmp64 = tmp53 * tmp63
        tmp65 = tmp64 * tmp23
        tmp66 = tmp47 + tmp65
        tmp68 = tmp67 + tmp66
        tl.store(in_out_ptr0 + (r0_2 + 4096*x3), tmp68, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 605, 4096), (4956160, 4096, 1))
    assert_size_stride(arg1_1, (2, 605, 4096), (4956160, 4096, 1))
    assert_size_stride(arg2_1, (4096, ), (1, ))
    assert_size_stride(arg3_1, (2, 605, 4096), (2478080, 4096, 1))
    assert_size_stride(arg4_1, (2, 605, 4096), (2478080, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((2, 605, 4096), (2478080, 4096, 1), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [float_1, square, mean, add_1, r0, truediv, float_2, square_1, mean_1, add_2, r1, truediv_1, add_3, inverse_mean, float_3, add, weighted, mul_6, add_5, xbar, mul_1, add_4, mul_2, inverse_secant, mul_5, coefficient, mul_7, mul_8, sum_1, mul_9, add_6, add_7], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.sqrt, aten.reciprocal, aten.mul, aten.div, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_0.run(buf4, arg0_1, arg1_1, arg3_1, arg2_1, arg4_1, 1210, 4096, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 605, 4096), (4956160, 4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2, 605, 4096), (4956160, 4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2, 605, 4096), (2478080, 4096, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 605, 4096), (2478080, 4096, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
