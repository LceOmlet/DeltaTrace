
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_max_exp_ge_gt_mul_sub_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 20971520, 'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_max_exp_ge_gt_mul_sub_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    x2 = xindex
    tmp28 = tl.load(in_ptr1 + (y0 + 64*x2 + 4096*y1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (y0 + 64*x2 + 4096*y1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (x2 + 64*y3), xmask, eviction_policy='evict_last')
    tmp0 = (y3 % 640)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 605, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(32*(((y0 + 64*y1) % 640)) + 38720*((y0 + 64*y1) // 20480) + ((((y0 + 64*y1) // 640) % 32)), [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 640, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (tl.broadcast_to(19328 + 38720*(y3 // 20480) + (((y3 // 640) % 32)), [YBLOCK, XBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = ((x2 + 64*y1) % 640)
    tmp12 = tmp11 >= tmp1
    tmp13 = tmp11 < tmp3
    tmp14 = tl.load(in_ptr0 + (32*(((x2 + 64*y1) % 640)) + 38720*((x2 + 64*y1) // 20480) + ((((x2 + 64*y1) // 640) % 32))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp11 >= tmp3
    tmp16 = tmp11 < tmp7
    tmp17 = tl.load(in_ptr0 + (19328 + 38720*((x2 + 64*y1) // 20480) + ((((x2 + 64*y1) // 640) % 32))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp13, tmp14, tmp17)
    tmp19 = tmp10 - tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tl_math.exp(tmp21)
    tmp23 = y0
    tmp24 = x2
    tmp25 = tmp23 >= tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp22 * tmp26
    tmp29 = tmp28 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp32 = tmp23 > tmp24
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp27 * tmp33
    tmp35 = tmp31 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp38 = tmp37 * tmp34
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr1 + (x2 + 64*y3), tmp30, xmask)
    tl.store(out_ptr2 + (x2 + 64*y3), tmp36, xmask)
    tl.store(out_ptr3 + (x2 + 64*y3), tmp39, xmask)
