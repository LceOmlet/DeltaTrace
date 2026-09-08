
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128860160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4956160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x0 = (xindex % 256)
    x3 = xindex // 2478080
    x4 = ((xindex // 256) % 9680)
    x1 = ((xindex // 256) % 16)
    x2 = ((xindex // 4096) % 605)
    x6 = (xindex % 2478080)
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + 512*x4 + 9912320*x3), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x6 + 4956160*x3), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (256 + x0 + 512*x4 + 9912320*x3), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 * tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp1.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = 0.0
    tmp13 = tmp11 != tmp12
    tmp14 = tl.sigmoid(tmp9)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp3 - tmp15
    tmp17 = 1.0
    tmp18 = tl.where(tmp13, tmp11, tmp17)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.sigmoid(tmp10)
    tmp21 = tmp17 - tmp20
    tmp22 = tmp20 * tmp21
    tmp23 = tl.where(tmp13, tmp19, tmp22)
    tmp24 = tmp7 * tmp23
    tl.store(out_ptr0 + (x0 + 256*x2 + 154880*x1 + 2478080*x3), tmp4, None)
    tl.store(out_ptr1 + (x5), tmp24, None)
