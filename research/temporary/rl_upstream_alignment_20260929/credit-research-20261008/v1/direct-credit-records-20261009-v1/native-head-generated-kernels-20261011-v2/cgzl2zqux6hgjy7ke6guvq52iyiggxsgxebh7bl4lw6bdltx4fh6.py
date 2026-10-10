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


# kernel path: /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/cache/inductor/xs/cxsxdnuskbaadgeknoor7j3y3rtewglwsc2kqddyoxjrrq645wpy.py
# Topologically Sorted Source Nodes: [log_probs, gather, logits_1, logsumexp, probs, mul, sum_1, entropy], Original ATen: [aten._log_softmax, aten.gather, aten._to_copy, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
# Source node to ATen node mapping:
#   entropy => sub_23
#   gather => gather
#   log_probs => exp_1, log, sub_12, sum_2
#   logits_1 => convert_element_type_2
#   logsumexp => abs_1, add_26, amax_2, eq_21, exp_2, full_default, log_1, sub_18, sum_3, where
#   mul => mul_17
#   probs => div_1, exp, sum_1
#   sum_1 => sum_4
# Graph fragment:
#   %convert_element_type_default_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %ge_scalar : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type_default, 0), kwargs = {})
#   %scalar_tensor_default : [num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (1,), kwargs = {dtype: torch.float32, device: cuda:0, pin_memory: False})
#   %neg_default : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%scalar_tensor_default,), kwargs = {})
#   %where_self : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_scalar, %scalar_tensor_default, %neg_default), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %where_self), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %convert_element_type_default), kwargs = {})
#   %div_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, %mul_tensor_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor_1, %log), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_12, -1, %unsqueeze), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, %convert_element_type_default), kwargs = {})
#   %convert_element_type_2 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_tensor, torch.float32), kwargs = {})
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
#   %convert_element_type_default_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %ge_scalar_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type_default, 0), kwargs = {})
#   %scalar_tensor_default_1 : [num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (1,), kwargs = {dtype: torch.float32, device: cuda:0, pin_memory: False})
#   %neg_default_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%scalar_tensor_default_1,), kwargs = {})
#   %where_self_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_scalar_1, %scalar_tensor_default_1, %neg_default_1), kwargs = {})
#   %mul_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_3, %where_self_1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_2, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_2, %amax_default_1), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self_1, %convert_element_type_default), kwargs = {})
#   %div_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, %mul_tensor_3), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %convert_element_type_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [-1]), kwargs = {dtype: torch.float32})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_26, %sum_4), kwargs = {})
triton_red_fused__log_softmax__softmax__to_copy_gather_logsumexp_mul_sub_sum_0 = async_compile.triton('triton_red_fused__log_softmax__softmax__to_copy_gather_logsumexp_mul_sub_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': 'fp64', 'in_ptr2': '*i64', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__softmax__to_copy_gather_logsumexp_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 7, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__softmax__to_copy_gather_logsumexp_mul_sub_sum_0(in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2 = in_ptr1
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.0
        tmp5 = tmp3 >= tmp4
        tmp6 = 1.0
        tmp7 = -1.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp2.to(tl.float32)
        tmp16 = 0.0
        tmp17 = tmp15 >= tmp16
        tmp18 = 1.0
        tmp19 = -1.0
        tmp20 = tl.where(tmp17, tmp18, tmp19)
        tmp21 = tmp14 * tmp20
        tmp22 = tmp21 - tmp11
        tmp23 = tmp20 * tmp15
        tmp24 = (tmp22 / tmp23)
        tmp25 = tl_math.exp(tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
        tmp29 = tmp15.to(tl.float32)
        tmp30 = (tmp13 / tmp29)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    _tmp48 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp57 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp35 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp2.to(tl.float32)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = (tmp35 / tmp37)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tl_math.abs(tmp33)
        tmp41 = float("inf")
        tmp42 = tmp40 == tmp41
        tmp43 = 0.0
        tmp44 = tl.where(tmp42, tmp43, tmp33)
        tmp45 = tmp39 - tmp44
        tmp46 = tl_math.exp(tmp45)
        tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
        tmp49 = _tmp48 + tmp47
        _tmp48 = tl.where(r0_mask & xmask, tmp49, _tmp48)
        tmp50 = tmp35.to(tl.float32)
        tmp51 = tmp36 >= tmp43
        tmp52 = 1.0
        tmp53 = -1.0
        tmp54 = tl.where(tmp51, tmp52, tmp53)
        tmp55 = tmp50 * tmp54
        tmp56 = tl.broadcast_to(tmp55, [XBLOCK, R0_BLOCK])
        tmp58 = triton_helpers.maximum(_tmp57, tmp56)
        _tmp57 = tl.where(r0_mask & xmask, tmp58, _tmp57)
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tmp57 = triton_helpers.max2(_tmp57, 1)[:, None]
    _tmp73 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp59 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp60 = tmp59.to(tl.float32)
        tmp61 = tmp2.to(tl.float32)
        tmp62 = 0.0
        tmp63 = tmp61 >= tmp62
        tmp64 = 1.0
        tmp65 = -1.0
        tmp66 = tl.where(tmp63, tmp64, tmp65)
        tmp67 = tmp60 * tmp66
        tmp68 = tmp67 - tmp57
        tmp69 = tmp66 * tmp61
        tmp70 = (tmp68 / tmp69)
        tmp71 = tl_math.exp(tmp70)
        tmp72 = tl.broadcast_to(tmp71, [XBLOCK, R0_BLOCK])
        tmp74 = _tmp73 + tmp72
        _tmp73 = tl.where(r0_mask & xmask, tmp74, _tmp73)
    tmp73 = tl.sum(_tmp73, 1)[:, None]
    _tmp94 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp75 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp76 = tmp75.to(tl.float32)
        tmp77 = tmp2.to(tl.float32)
        tmp78 = 0.0
        tmp79 = tmp77 >= tmp78
        tmp80 = 1.0
        tmp81 = -1.0
        tmp82 = tl.where(tmp79, tmp80, tmp81)
        tmp83 = tmp76 * tmp82
        tmp84 = tmp83 - tmp57
        tmp85 = tmp82 * tmp77
        tmp86 = (tmp84 / tmp85)
        tmp87 = tl_math.exp(tmp86)
        tmp88 = (tmp87 / tmp73)
        tmp89 = tmp77.to(tl.float32)
        tmp90 = (tmp75 / tmp89)
        tmp91 = tmp90.to(tl.float32)
        tmp92 = tmp88 * tmp91
        tmp93 = tl.broadcast_to(tmp92, [XBLOCK, R0_BLOCK])
        tmp95 = _tmp94 + tmp93
        _tmp94 = tl.where(r0_mask & xmask, tmp95, _tmp94)
    tmp94 = tl.sum(_tmp94, 1)[:, None]
    tmp96 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp97 = ks0
    tmp98 = tmp96 + tmp97
    tmp99 = tmp96 < 0
    tmp100 = tl.where(tmp99, tmp98, tmp96)
    tmp101 = tl.load(in_ptr0 + (tmp100 + ks0*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tmp2.to(tl.float32)
    tmp104 = 0.0
    tmp105 = tmp103 >= tmp104
    tmp106 = 1.0
    tmp107 = -1.0
    tmp108 = tl.where(tmp105, tmp106, tmp107)
    tmp109 = tmp102 * tmp108
    tmp110 = tmp109 - tmp11
    tmp111 = tmp108 * tmp103
    tmp112 = (tmp110 / tmp111)
    tmp113 = tl_math.log(tmp27)
    tmp114 = tmp112 - tmp113
    tmp115 = tl_math.log(tmp48)
    tmp116 = tl_math.abs(tmp33)
    tmp117 = float("inf")
    tmp118 = tmp116 == tmp117
    tmp119 = tl.where(tmp118, tmp104, tmp33)
    tmp120 = tmp115 + tmp119
    tmp121 = tmp120 - tmp94
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp114, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp121, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    s15 = arg0_1
    s16 = arg1_1
    s98 = arg3_1
    assert_size_stride(arg2_1, (s15, s16), (s16, 1))
    assert_size_stride(arg4_1, (s98, s16), (s16, 1))
    assert_size_stride(arg5_1, (), ())
    assert_size_stride(arg6_1, (s98, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s98, s15), (s15, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(arg4_1, reinterpret_tensor(arg2_1, (s16, s15), (1, s16), 0), out=buf0)
        del arg2_1
        del arg4_1
        buf1 = empty_strided_cuda((s98, 1), (1, s98), torch.float32)
        buf5 = empty_strided_cuda((s98, ), (1, ), torch.float32)
        buf3 = reinterpret_tensor(buf1, (s98, 1), (1, 1), 0); del buf1  # reuse
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [log_probs, gather, logits_1, logsumexp, probs, mul, sum_1, entropy], Original ATen: [aten._log_softmax, aten.gather, aten._to_copy, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__softmax__to_copy_gather_logsumexp_mul_sub_sum_0.run(buf3, buf9, buf0, arg5_1.item(), arg6_1, s15, s98, s15, stream=stream0)
        del arg5_1
        del arg6_1
        del buf0
    return (reinterpret_tensor(buf3, (s98, ), (1, ), 0), buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 248320
    arg1_1 = 4096
    arg2_1 = rand_strided((248320, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = 512
    arg4_1 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
