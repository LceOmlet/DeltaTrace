
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sub_sum_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 117207040}}
)
@triton.jit
def triton_red_fused__to_copy_mul_sub_sum_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 472
    r0_numel = 31040
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    x2 = (xindex % 8)
    x3 = xindex // 8
    tmp4 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 31040*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 31040*x2 + 496640*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 31040*x2 + 496640*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 - tmp12
        tmp14 = tl_math.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
        tmp18 = tmp11 - tmp3
        tmp19 = tmp0 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
        tl.store(out_ptr0 + (r0_1 + 31040*x0), tmp1, r0_mask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
