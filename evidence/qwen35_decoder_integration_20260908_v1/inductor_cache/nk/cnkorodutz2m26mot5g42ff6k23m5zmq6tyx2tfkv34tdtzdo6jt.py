
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 60712960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4956160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x4 = xindex // 256
    x1 = ((xindex // 256) % 605)
    x3 = xindex // 2478080
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (64*x1 + 77440*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = x0
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 32, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp14 & tmp4
    tmp16 = tl.load(in_ptr0 + (32 + 256*x4 + (x0)), tmp15, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (32 + 64*x1 + 77440*x3 + (x0)), tmp15, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = -tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp15, tmp21, tmp22)
    tmp24 = tmp10 >= tmp13
    tmp25 = tl.full([1], 64, tl.int64)
    tmp26 = tmp10 < tmp25
    tmp27 = tmp24 & tmp4
    tmp28 = tl.load(in_ptr0 + (256*x4 + ((-32) + (x0))), tmp27, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (64*x1 + 77440*x3 + ((-32) + (x0))), tmp27, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp27, tmp32, tmp33)
    tmp35 = tl.where(tmp14, tmp23, tmp34)
    tmp36 = tmp9 - tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp4, tmp36, tmp37)
    tmp39 = tmp0 >= tmp3
    tmp40 = tl.full([1], 256, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr0 + (64 + 256*x4 + ((-64) + x0)), tmp39, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp39, tmp43, tmp44)
    tmp46 = tl.where(tmp4, tmp38, tmp45)
    tl.store(out_ptr0 + (x5), tmp46, None)
