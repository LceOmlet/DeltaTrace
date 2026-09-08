
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 356843520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_ne_ones_like_rsub_sigmoid_sub_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14868480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    x2 = (xindex % 7434240)
    x3 = xindex // 7434240
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 12288*((x1 % 605)) + 14868480*(x1 // 605)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 12288*((x1 % 605)) + 14868480*(x1 // 605)), None).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp19 = tl.load(in_ptr6 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2 + 14868480*x3), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp15 = tmp0 * tmp14
    tmp16 = tmp15 * tmp7
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 - tmp20
    tmp22 = 0.0
    tmp23 = tmp21 != tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 - tmp27
    tmp29 = 1.0
    tmp30 = tl.where(tmp23, tmp21, tmp29)
    tmp31 = (tmp28 / tmp30)
    tmp32 = tl.sigmoid(tmp20)
    tmp33 = tmp29 - tmp32
    tmp34 = tmp20 * tmp33
    tmp35 = tmp34 + tmp29
    tmp36 = tmp32 * tmp35
    tmp37 = tl.where(tmp23, tmp31, tmp36)
    tmp38 = tmp16 * tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp9, None)
    tl.store(out_ptr2 + (x4), tmp39, None)
