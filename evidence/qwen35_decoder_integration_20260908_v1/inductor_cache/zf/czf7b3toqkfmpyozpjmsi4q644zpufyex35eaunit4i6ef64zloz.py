
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 19825152}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_pow_reciprocal_sqrt_sum_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr4, xnumel, r0_numel):
    xnumel = 4840
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 2420)
    x1 = xindex // 2420
    x3 = (xindex % 4)
    x4 = ((xindex // 4) % 605)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 256*x0 + 1239040*x1), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (r0_2 + 256*x0 + 1239040*x1), None).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r0_2 + 256*x4 + 154880*x3 + 619520*x1), None)
    tmp16 = tl.load(in_ptr3 + (r0_2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp1 + tmp7
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp15 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 256.0
    tmp26 = (tmp5 / tmp25)
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1], 1, tl.int32)
    tmp31 = (tmp30 / tmp29)
    tmp32 = tmp31 * tmp18
    tmp33 = (tmp11 / tmp25)
    tmp34 = tmp33 + tmp27
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = (tmp30 / tmp35)
    tmp37 = tmp36 * tmp18
    tmp38 = tmp32 + tmp37
    tmp39 = tmp38 * tmp13
    tmp40 = tmp39 * tmp20
    tmp41 = tmp29 * tmp35
    tmp42 = tmp29 + tmp35
    tmp43 = tmp41 * tmp42
    tmp44 = (tmp30 / tmp43)
    tmp45 = -1.0
    tmp46 = tmp44 * tmp45
    tmp47 = 2.0
    tmp48 = tmp46 * tmp47
    tmp49 = 0.00390625
    tmp50 = tmp48 * tmp49
    tmp51 = tmp14 * tmp50
    tmp52 = tmp51 * tmp24
    tmp53 = tmp40 + tmp52
    tmp54 = tmp53.to(tl.float32)
    tl.store(out_ptr4 + (r0_2 + 256*x5), tmp54, None)
