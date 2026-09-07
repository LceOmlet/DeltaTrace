
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_constant_pad_nd_mul_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3244032, 'r0_': 12582912}}
)
@triton.jit
def triton_red_fused__to_copy_constant_pad_nd_mul_sum_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 32)
    x2 = xindex // 6144
    x4 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp5 = tl.load(in_ptr1 + (r0_3 + 128*x4), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 129, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (528384 + x0 + 129*r0_3 + 16512*x1 + 1056768*x2), r0_mask & tmp2, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tl.store(out_ptr0 + (r0_3 + 128*x4), tmp3, r0_mask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x4), tmp8, None)
