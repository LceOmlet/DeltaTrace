# AOT ID: ['9_inference']
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


# kernel path: /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/cache/inductor/4e/c4el6gli7ymnf4jpzff3atrxui6lp7aflg44wyyxmhnwjha43cno.py
# Topologically Sorted Source Nodes: [one_hot_input, probs, sub, mul, to_2, dlogits, log_probs, logits, logits_1, logsumexp, mul_1, sum_1, add_1, mul_2, neg, mul_3, to_3, dlogits_1, dlogits_2], Original ATen: [aten.scatter, aten._softmax, aten.sub, aten.mul, aten._to_copy, aten.add, aten._log_softmax, aten.div, aten.logsumexp, aten.sum, aten.neg]
# Source node to ATen node mapping:
#   add_1 => add_59
#   dlogits => add_39
#   dlogits_1 => add_84
#   dlogits_2 => div_2
#   log_probs => exp_1, log, sub_17, sum_2
#   logits => div
#   logits_1 => convert_element_type_2
#   logsumexp => abs_1, amax_2, eq_21, exp_2, full_default_1, sub_19, sum_3, where
#   mul => mul_20
#   mul_1 => mul_30
#   mul_2 => mul_38
#   mul_3 => mul_43
#   neg => neg
#   one_hot_input => scatter_upon_const_tensor
#   probs => div_1, exp, sum_1
#   sub => sub_11
#   sum_1 => sum_4
#   to_2 => convert_element_type_3
#   to_3 => convert_element_type_4
# Graph fragment:
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [%arg1_1, 248320], background_val: 0, dtype: torch.float32, dim: -1, selector: %unsqueeze, val: 1})
#   %convert_element_type_default_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%scatter_upon_const_tensor, %div_1), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %sub_11), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_20, torch.bfloat16), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 0), kwargs = {})
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor, %log), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 1.0), kwargs = {})
#   %convert_element_type_2 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %amax_2 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_2,), kwargs = {})
#   %eq_21 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_21, %full_default_1, %amax_2), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %where), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_19,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1]), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %convert_element_type_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_30, [-1]), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_17, %unsqueeze_2), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %add_59), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%unsqueeze_3,), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %neg), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_43, torch.bfloat16), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %convert_element_type_4), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_84, 1.0), kwargs = {})
triton_red_fused__log_softmax__softmax__to_copy_add_div_logsumexp_mul_neg_scatter_sub_sum_0 = async_compile.triton('triton_red_fused__log_softmax__softmax__to_copy_add_div_logsumexp_mul_neg_scatter_sub_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__softmax__to_copy_add_div_logsumexp_mul_neg_scatter_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 7, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__softmax__to_copy_add_div_logsumexp_mul_neg_scatter_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 248320
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 248320*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 1.0
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_out_ptr0 + (r0_1 + 248320*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10 - tmp5
        tmp12 = tmp11 * tmp9
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp36 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_out_ptr0 + (r0_1 + 248320*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = 1.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp5
        tmp22 = tmp21 * tmp19
        tmp23 = tl_math.exp(tmp22)
        tmp24 = (tmp23 / tmp15)
        tmp25 = tmp17.to(tl.bfloat16)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 * tmp19
        tmp28 = tmp27.to(tl.bfloat16)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp24 * tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
        tmp35 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp37 = triton_helpers.maximum(_tmp36, tmp35)
        _tmp36 = tl.where(r0_mask & xmask, tmp37, _tmp36)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp36 = triton_helpers.max2(_tmp36, 1)[:, None]
    _tmp46 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp55 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp38 = tl.load(in_out_ptr0 + (r0_1 + 248320*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = 1.0
        tmp41 = tmp39 * tmp40
        tmp42 = tmp41 - tmp36
        tmp43 = tmp42 * tmp40
        tmp44 = tl_math.exp(tmp43)
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
        tmp47 = _tmp46 + tmp45
        _tmp46 = tl.where(r0_mask & xmask, tmp47, _tmp46)
        tmp48 = tmp38.to(tl.bfloat16)
        tmp49 = tmp48.to(tl.float32)
        tmp50 = tmp49 * tmp40
        tmp51 = tmp50.to(tl.bfloat16)
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tl.broadcast_to(tmp53, [XBLOCK, R0_BLOCK])
        tmp56 = triton_helpers.maximum(_tmp55, tmp54)
        _tmp55 = tl.where(r0_mask & xmask, tmp56, _tmp55)
    tmp46 = tl.sum(_tmp46, 1)[:, None]
    tmp55 = triton_helpers.max2(_tmp55, 1)[:, None]
    _tmp73 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp57 = tl.load(in_out_ptr0 + (r0_1 + 248320*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tmp57.to(tl.bfloat16)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = 1.0
        tmp61 = tmp59 * tmp60
        tmp62 = tmp61.to(tl.bfloat16)
        tmp63 = tmp62.to(tl.float32)
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tl_math.abs(tmp55)
        tmp66 = float("inf")
        tmp67 = tmp65 == tmp66
        tmp68 = 0.0
        tmp69 = tl.where(tmp67, tmp68, tmp55)
        tmp70 = tmp64 - tmp69
        tmp71 = tl_math.exp(tmp70)
        tmp72 = tl.broadcast_to(tmp71, [XBLOCK, R0_BLOCK])
        tmp74 = _tmp73 + tmp72
        _tmp73 = tl.where(r0_mask & xmask, tmp74, _tmp73)
    tmp73 = tl.sum(_tmp73, 1)[:, None]
    tmp97 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp75 = tl.load(in_out_ptr0 + (r0_1 + 248320*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp76 = tmp75.to(tl.float32)
        tmp77 = 1.0
        tmp78 = tmp76 * tmp77
        tmp79 = tmp78 - tmp5
        tmp80 = tmp79 * tmp77
        tmp81 = tl_math.exp(tmp80)
        tmp82 = (tmp81 / tmp15)
        tmp83 = tmp78 - tmp36
        tmp84 = tmp83 * tmp77
        tmp85 = tl_math.log(tmp46)
        tmp86 = tmp84 - tmp85
        tmp87 = tl_math.log(tmp73)
        tmp88 = tl_math.abs(tmp55)
        tmp89 = float("inf")
        tmp90 = tmp88 == tmp89
        tmp91 = 0.0
        tmp92 = tl.where(tmp90, tmp91, tmp55)
        tmp93 = tmp87 + tmp92
        tmp94 = tmp93 - tmp33
        tmp95 = tmp86 + tmp94
        tmp96 = tmp82 * tmp95
        tmp98 = -tmp97
        tmp99 = tmp96 * tmp98
        tmp102 = r0_1
        tmp103 = tmp101 == tmp102
        tmp104 = tl.where(tmp103, tmp77, tmp91)
        tmp105 = tmp104 - tmp82
        tmp106 = tmp100 * tmp105
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tmp107.to(tl.bfloat16)
        tmp109 = tmp108.to(tl.float32)
        tmp110 = tmp109 + tmp91
        tmp111 = tmp110.to(tl.bfloat16)
        tmp112 = tmp111.to(tl.float32)
        tmp113 = tmp112.to(tl.bfloat16)
        tmp114 = tmp113.to(tl.float32)
        tmp115 = tmp99.to(tl.float32)
        tmp116 = tmp115.to(tl.bfloat16)
        tmp117 = tmp116.to(tl.float32)
        tmp118 = tmp114 + tmp117
        tmp119 = tmp118.to(tl.bfloat16)
        tmp120 = tmp119.to(tl.float32)
        tmp121 = tmp120.to(tl.bfloat16)
        tmp122 = tmp121.to(tl.float32)
        tmp123 = tmp122 * tmp77
        tmp124 = tmp123.to(tl.bfloat16)
        tmp125 = tmp124.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 248320*x0), tmp125, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    s47 = arg1_1
    assert_size_stride(arg0_1, (248320, 4096), (4096, 1))
    assert_size_stride(arg2_1, (s47, 4096), (4096, 1))
    assert_size_stride(arg3_1, (s47, ), (1, ))
    assert_size_stride(arg4_1, (s47, ), (1, ))
    assert_size_stride(arg5_1, (s47, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s47, 248320), (248320, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(arg2_1, reinterpret_tensor(arg0_1, (4096, 248320), (1, 4096), 0), out=buf0)
        buf9 = buf0; del buf0  # reuse
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [one_hot_input, probs, sub, mul, to_2, dlogits, log_probs, logits, logits_1, logsumexp, mul_1, sum_1, add_1, mul_2, neg, mul_3, to_3, dlogits_1, dlogits_2], Original ATen: [aten.scatter, aten._softmax, aten.sub, aten.mul, aten._to_copy, aten.add, aten._log_softmax, aten.div, aten.logsumexp, aten.sum, aten.neg]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__softmax__to_copy_add_div_logsumexp_mul_neg_scatter_sub_sum_0.run(buf10, arg5_1, arg3_1, arg4_1, s47, 248320, stream=stream0)
        del arg3_1
        del arg4_1
        del arg5_1
        buf11 = empty_strided_cuda((s47, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [dhidden_states], Original ATen: [aten.mm]
        extern_kernels.mm(buf10, arg0_1, out=buf11)
        del arg0_1
        buf12 = empty_strided_cuda((248320, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [dvocab_weights], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (248320, s47), (1, 248320), 0), arg2_1, out=buf12)
        del arg2_1
        del buf10
    return (buf11, buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((248320, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 512
    arg2_1 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
