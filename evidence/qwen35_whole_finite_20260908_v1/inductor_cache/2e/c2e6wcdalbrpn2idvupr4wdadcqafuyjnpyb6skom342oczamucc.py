
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_all_eq_ge_mul_ne_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1792, 'r0_': 58603776}}
)
@triton.jit
def triton_red_fused_abs_all_eq_ge_mul_ne_7(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 448
    r0_numel = 32703
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + 32703*x0
        tmp1 = tl.full([1, 1], 14650880, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + 32703*x0) % 14650880)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp3 == tmp3
        tmp5 = tl_math.abs(tmp3)
        tmp6 = float("inf")
        tmp7 = tmp5 != tmp6
        tmp8 = tmp4 & tmp7
        tmp9 = tmp8 == 0
        tmp10 = tl.full(tmp9.shape, False, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 | tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
        tmp15 = 0.0
        tmp16 = tmp3 >= tmp15
        tmp17 = tmp16 == 0
        tmp18 = tl.full(tmp17.shape, False, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 | tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp13 = triton_helpers.any(_tmp13.to(tl.int8), 1)[:, None].to(tl.int1)
    tmp21 = triton_helpers.any(_tmp21.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp21, xmask)
