
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 929280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 605)
    x2 = xindex // 19360
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 640*x0 + 20480*x2), xmask, eviction_policy='evict_last')
    tmp1 = x1
    tmp2 = tl.full([1], 605, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x0 + 32*x1 + 38720*x2 + 38720*(triton_helpers.div_floor_integer(64*(x1 // 64) + 640*x0 + ((x1 % 64)),  20480)) + (triton_helpers.div_floor_integer(64*(x1 // 64) + ((x1 % 64)),  640))), tmp3 & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (19360 + x0 + 32*x1 + 38720*x2 + 38720*(triton_helpers.div_floor_integer(64*(x1 // 64) + 640*x0 + ((x1 % 64)),  20480)) + (triton_helpers.div_floor_integer(64*(x1 // 64) + ((x1 % 64)),  640))), tmp3 & xmask, other=0.0)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp0 * tmp7
    tmp9 = tmp5 - tmp4
    tmp10 = tl_math.abs(tmp9)
    tmp11 = 0.0
    tmp12 = tmp10 == tmp11
    tmp13 = -tmp10
    tmp14 = libdevice.expm1(tmp13)
    tmp15 = -tmp14
    tmp16 = 1.0
    tmp17 = tl.where(tmp12, tmp16, tmp10)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tl.where(tmp12, tmp16, tmp18)
    tmp20 = tmp8 * tmp19
    tl.store(out_ptr0 + (x3), tmp0, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
