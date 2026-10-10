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


# kernel path: /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/cache/inductor/vz/cvzcdkzy3xk6mvxwiliskdreyecgl7ktxeibuaivcvsyleidwdb4.py
# Topologically Sorted Source Nodes: [log_probs, gather, logits, logits_1, logsumexp, probs, mul, sum_1, entropy], Original ATen: [aten._log_softmax, aten.gather, aten.div, aten._to_copy, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
# Source node to ATen node mapping:
#   entropy => sub_23
#   gather => gather
#   log_probs => exp_1, log, sub_12, sum_2
#   logits => div
#   logits_1 => convert_element_type_2
#   logsumexp => abs_1, add_26, amax_2, eq_21, exp_2, full_default, log_1, sub_18, sum_3, where
#   mul => mul_17
#   probs => div_1, exp, sum_1
#   sum_1 => sum_4
# Graph fragment:
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor, %log), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_12, -1, %unsqueeze), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 1.0), kwargs = {})
#   %convert_element_type_2 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %amax_2 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_2,), kwargs = {})
#   %eq_21 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_21, %full_default, %amax_2), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %where), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_18,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1]), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_3,), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_1, %squeeze_1), kwargs = {})
#   %convert_element_type_default_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %convert_element_type_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [-1]), kwargs = {dtype: torch.float32})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_26, %sum_4), kwargs = {})
triton_red_fused__log_softmax__softmax__to_copy_div_gather_logsumexp_mul_sub_sum_0 = async_compile.triton('triton_red_fused__log_softmax__softmax__to_copy_div_gather_logsumexp_mul_sub_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__softmax__to_copy_div_gather_logsumexp_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 7, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__softmax__to_copy_div_gather_logsumexp_mul_sub_sum_0(in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 1.0
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10 - tmp5
        tmp12 = tmp11 * tmp9
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
        tmp17 = tmp7.to(tl.bfloat16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 * tmp9
        tmp20 = tmp19.to(tl.bfloat16)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = triton_helpers.maximum(_tmp24, tmp23)
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp24 = triton_helpers.max2(_tmp24, 1)[:, None]
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp47 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp26 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tmp26.to(tl.bfloat16)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = 1.0
        tmp30 = tmp28 * tmp29
        tmp31 = tmp30.to(tl.bfloat16)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl_math.abs(tmp24)
        tmp35 = float("inf")
        tmp36 = tmp34 == tmp35
        tmp37 = 0.0
        tmp38 = tl.where(tmp36, tmp37, tmp24)
        tmp39 = tmp33 - tmp38
        tmp40 = tl_math.exp(tmp39)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask & xmask, tmp43, _tmp42)
        tmp44 = tmp26.to(tl.float32)
        tmp45 = tmp44 * tmp29
        tmp46 = tl.broadcast_to(tmp45, [XBLOCK, R0_BLOCK])
        tmp48 = triton_helpers.maximum(_tmp47, tmp46)
        _tmp47 = tl.where(r0_mask & xmask, tmp48, _tmp47)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp47 = triton_helpers.max2(_tmp47, 1)[:, None]
    _tmp57 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp49 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp50 = tmp49.to(tl.float32)
        tmp51 = 1.0
        tmp52 = tmp50 * tmp51
        tmp53 = tmp52 - tmp47
        tmp54 = tmp53 * tmp51
        tmp55 = tl_math.exp(tmp54)
        tmp56 = tl.broadcast_to(tmp55, [XBLOCK, R0_BLOCK])
        tmp58 = _tmp57 + tmp56
        _tmp57 = tl.where(r0_mask & xmask, tmp58, _tmp57)
    tmp57 = tl.sum(_tmp57, 1)[:, None]
    _tmp75 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp59 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp60 = tmp59.to(tl.float32)
        tmp61 = 1.0
        tmp62 = tmp60 * tmp61
        tmp63 = tmp62 - tmp47
        tmp64 = tmp63 * tmp61
        tmp65 = tl_math.exp(tmp64)
        tmp66 = (tmp65 / tmp57)
        tmp67 = tmp59.to(tl.bfloat16)
        tmp68 = tmp67.to(tl.float32)
        tmp69 = tmp68 * tmp61
        tmp70 = tmp69.to(tl.bfloat16)
        tmp71 = tmp70.to(tl.float32)
        tmp72 = tmp71.to(tl.float32)
        tmp73 = tmp66 * tmp72
        tmp74 = tl.broadcast_to(tmp73, [XBLOCK, R0_BLOCK])
        tmp76 = _tmp75 + tmp74
        _tmp75 = tl.where(r0_mask & xmask, tmp76, _tmp75)
    tmp75 = tl.sum(_tmp75, 1)[:, None]
    tmp77 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp78 = ks0
    tmp79 = tmp77 + tmp78
    tmp80 = tmp77 < 0
    tmp81 = tl.where(tmp80, tmp79, tmp77)
    tmp82 = tl.load(in_ptr0 + (tmp81 + ks0*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp83 = tmp82.to(tl.float32)
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp85 - tmp5
    tmp87 = tmp86 * tmp84
    tmp88 = tl_math.log(tmp15)
    tmp89 = tmp87 - tmp88
    tmp90 = tl_math.log(tmp42)
    tmp91 = tl_math.abs(tmp24)
    tmp92 = float("inf")
    tmp93 = tmp91 == tmp92
    tmp94 = 0.0
    tmp95 = tl.where(tmp93, tmp94, tmp24)
    tmp96 = tmp90 + tmp95
    tmp97 = tmp96 - tmp75
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp89, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp97, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    s15 = arg0_1
    s16 = arg1_1
    s47 = arg3_1
    assert_size_stride(arg2_1, (s15, s16), (s16, 1))
    assert_size_stride(arg4_1, (s47, s16), (s16, 1))
    assert_size_stride(arg5_1, (s47, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s47, s15), (s15, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(arg4_1, reinterpret_tensor(arg2_1, (s16, s15), (1, s16), 0), out=buf0)
        del arg2_1
        del arg4_1
        buf1 = empty_strided_cuda((s47, 1), (1, s47), torch.float32)
        buf5 = empty_strided_cuda((s47, ), (1, ), torch.float32)
        buf3 = reinterpret_tensor(buf1, (s47, 1), (1, 1), 0); del buf1  # reuse
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [log_probs, gather, logits, logits_1, logsumexp, probs, mul, sum_1, entropy], Original ATen: [aten._log_softmax, aten.gather, aten.div, aten._to_copy, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__softmax__to_copy_div_gather_logsumexp_mul_sub_sum_0.run(buf3, buf9, buf0, arg5_1, s15, s47, s15, stream=stream0)
        del arg5_1
        del buf0
    return (reinterpret_tensor(buf3, (s47, ), (1, ), 0), buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 248320
    arg1_1 = 4096
    arg2_1 = rand_strided((248320, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = 512
    arg4_1 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
