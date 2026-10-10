# AOT ID: ['4_inference']
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


# kernel path: /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/cache/inductor/7a/c7ayslq72kl2xbogrqtkr2kww2s2eg37ghf5kzes2dzzawoolu7n.py
# Topologically Sorted Source Nodes: [log_probs_1, gather, logits, logsumexp, probs, mul, sum_1, entropy_1], Original ATen: [aten._log_softmax, aten.gather, aten.div, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
# Source node to ATen node mapping:
#   entropy_1 => sub_14
#   gather => gather
#   log_probs_1 => exp_1, log, sub_10, sum_2
#   logits => div
#   logsumexp => abs_1, add_23, amax_2, eq_18, exp_2, full_default_2, log_1, sub_12, sum_3, where
#   mul => mul_18
#   probs => div_1, exp, sum_1
#   sum_1 => sum_4
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor, %log), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_10, -1, %unsqueeze), kwargs = {})
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 1.0), kwargs = {})
#   %amax_2 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%div, [-1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_2,), kwargs = {})
#   %eq_18 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_18, %full_default_2, %amax_2), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %where), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_12,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1]), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_3,), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_1, %squeeze_1), kwargs = {})
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [-1]), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_23, %sum_4), kwargs = {})
triton_red_fused__log_softmax__softmax_div_gather_logsumexp_mul_sub_sum_0 = async_compile.triton('triton_red_fused__log_softmax__softmax_div_gather_logsumexp_mul_sub_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__softmax_div_gather_logsumexp_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 7, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__softmax_div_gather_logsumexp_mul_sub_sum_0(in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8 - tmp4
        tmp10 = tmp9 * tmp7
        tmp11 = tl_math.exp(tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp17 = triton_helpers.maximum(_tmp16, tmp15)
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp16 = triton_helpers.max2(_tmp16, 1)[:, None]
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp32 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp18 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = 1.0
        tmp20 = tmp18 * tmp19
        tmp21 = tl_math.abs(tmp16)
        tmp22 = float("inf")
        tmp23 = tmp21 == tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp23, tmp24, tmp16)
        tmp26 = tmp20 - tmp25
        tmp27 = tl_math.exp(tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask & xmask, tmp30, _tmp29)
        tmp31 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp33 = triton_helpers.maximum(_tmp32, tmp31)
        _tmp32 = tl.where(r0_mask & xmask, tmp33, _tmp32)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp32 = triton_helpers.max2(_tmp32, 1)[:, None]
    _tmp41 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp34 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = 1.0
        tmp36 = tmp34 * tmp35
        tmp37 = tmp36 - tmp32
        tmp38 = tmp37 * tmp35
        tmp39 = tl_math.exp(tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    _tmp52 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp43 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp44 = 1.0
        tmp45 = tmp43 * tmp44
        tmp46 = tmp45 - tmp32
        tmp47 = tmp46 * tmp44
        tmp48 = tl_math.exp(tmp47)
        tmp49 = (tmp48 / tmp41)
        tmp50 = tmp49 * tmp45
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(r0_mask & xmask, tmp53, _tmp52)
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    tmp54 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp55 = ks0
    tmp56 = tmp54 + tmp55
    tmp57 = tmp54 < 0
    tmp58 = tl.where(tmp57, tmp56, tmp54)
    tmp59 = tl.load(in_ptr0 + (tmp58 + ks0*x0), xmask, eviction_policy='evict_last')
    tmp60 = 1.0
    tmp61 = tmp59 * tmp60
    tmp62 = tmp61 - tmp4
    tmp63 = tmp62 * tmp60
    tmp64 = tl_math.log(tmp13)
    tmp65 = tmp63 - tmp64
    tmp66 = tl_math.log(tmp29)
    tmp67 = tl_math.abs(tmp16)
    tmp68 = float("inf")
    tmp69 = tmp67 == tmp68
    tmp70 = 0.0
    tmp71 = tl.where(tmp69, tmp70, tmp16)
    tmp72 = tmp66 + tmp71
    tmp73 = tmp72 - tmp52
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp65, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp73, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    s70 = arg0_1
    s98 = arg2_1
    s15 = arg4_1
    assert_size_stride(arg1_1, (1, s70), (s70, 1))
    assert_size_stride(arg3_1, (1, 256, s98), (256*s98, s98, 1))
    assert_size_stride(arg5_1, (s15, s98), (s98, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, s15), (s15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg3_1, (256, s98), (s98, 1), 0), reinterpret_tensor(arg5_1, (s98, s15), (1, s98), 0), out=buf0)
        del arg3_1
        del arg5_1
        buf1 = empty_strided_cuda((256, 1), (1, 256), torch.float32)
        buf5 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf3 = reinterpret_tensor(buf1, (256, 1), (1, 1), 0); del buf1  # reuse
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [log_probs_1, gather, logits, logsumexp, probs, mul, sum_1, entropy_1], Original ATen: [aten._log_softmax, aten.gather, aten.div, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__softmax_div_gather_logsumexp_mul_sub_sum_0.run(buf3, buf9, buf0, arg1_1, s15, 256, s15, stream=stream0)
        del arg1_1
        del buf0
    return (reinterpret_tensor(buf3, (1, 256), (256, 1), 0), reinterpret_tensor(buf9, (1, 256), (256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 256
    arg1_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = 4096
    arg3_1 = rand_strided((1, 256, 4096), (1048576, 4096, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = 248320
    arg5_1 = rand_strided((248320, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
