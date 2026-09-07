
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_max_exp_mul_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_max_exp_mul_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 64)
    x3 = xindex // 4096
    x2 = ((xindex // 64) % 64)
    x4 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 - tmp7
    tmp9 = ((x1 + 64*x3) % 192)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 129, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(in_ptr3 + (4128 + 32*(((x1 + 64*x3) % 192)) + 8256*((x1 + 64*x3) // 6144) + ((((x1 + 64*x3) // 192) % 32))), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp12
    tmp16 = tl.full([1], 192, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tl.load(in_ptr3 + (8224 + 8256*((x1 + 64*x3) // 6144) + ((((x1 + 64*x3) // 192) % 32))), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp13, tmp14, tmp18)
    tmp20 = ((x0 // 64) % 192)
    tmp21 = tmp20 >= tmp10
    tmp22 = tmp20 < tmp12
    tmp23 = tl.load(in_ptr3 + (4128 + 32*(((x2 + 64*x3) % 192)) + 8256*((x2 + 64*x3) // 6144) + ((((x2 + 64*x3) // 192) % 32))), tmp22, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp20 >= tmp12
    tmp25 = tmp20 < tmp16
    tmp26 = tl.load(in_ptr3 + (8224 + 8256*(x4 // 6144) + (((x4 // 192) % 32))), tmp24, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp22, tmp23, tmp26)
    tmp28 = tmp19 - tmp27
    tmp29 = 0.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tl_math.exp(tmp30)
    tmp32 = x1
    tmp33 = x2
    tmp34 = tmp32 >= tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp31 * tmp35
    tmp37 = tmp3 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp6 * tmp36
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp38, None)
    tl.store(out_ptr2 + (x0), tmp40, None)
