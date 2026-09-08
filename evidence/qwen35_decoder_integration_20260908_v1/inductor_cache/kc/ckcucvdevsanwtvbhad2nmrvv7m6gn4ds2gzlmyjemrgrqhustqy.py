
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
