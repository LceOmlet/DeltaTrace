
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2113536, 'x': 12681216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8256
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y1 = ((yindex // 32) % 129)
    y0 = (yindex % 32)
    y2 = yindex // 4128
    y4 = yindex
    tmp5 = tl.load(in_ptr1 + (x3 + 128*((y1 % 64)) + 8192*((x3 + 128*y1) // 8192) + 24576*y0 + 786432*y2), xmask & ymask, eviction_policy='evict_last')
    tmp0 = 64*((x3 + 128*y1) // 8192) + ((y1 % 64))
    tmp1 = tl.full([1, 1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4128 + y0 + 32*((y1 % 64)) + 2048*((x3 + 128*y1) // 8192) + 8256*y2 + 8256*(triton_helpers.div_floor_integer(64*((x3 + 128*y1) // 8192) + 192*y0 + ((y1 % 64)),  6144)) + (triton_helpers.div_floor_integer(64*((x3 + 128*y1) // 8192) + ((y1 % 64)),  192))), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x3 + 128*y4), tmp6, xmask & ymask)
