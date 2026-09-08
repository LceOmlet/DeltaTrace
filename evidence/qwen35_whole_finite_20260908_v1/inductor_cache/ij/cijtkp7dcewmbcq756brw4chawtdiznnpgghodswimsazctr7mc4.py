
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
