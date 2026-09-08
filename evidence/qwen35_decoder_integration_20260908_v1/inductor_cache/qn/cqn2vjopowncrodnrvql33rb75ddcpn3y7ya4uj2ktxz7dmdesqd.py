
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*bf16', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sub_sum_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 29, 'num_reduction': 4, 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1802240, 'r0_': 283115520}}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sub_sum_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 40960
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = (x2 % 640)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 605, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*(((x0 + 64*x1) % 640)) + 38720*((x0 + 64*x1) // 20480) + ((((x0 + 64*x1) // 640) % 32))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 640, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (19328 + 38720*(x2 // 20480) + (((x2 // 640) % 32))), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp44 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp11 = tl.load(in_ptr1 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr4 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr5 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr7 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr8 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp59 = tl.load(in_out_ptr0 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp61 = tl.load(in_ptr9 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr3 + (tl.broadcast_to(19360 + 32*((x2 % 640)) + 38720*(x2 // 20480) + (((x2 // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask, tmp26, _tmp25)
        tmp27 = tmp12 * tmp20
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask, tmp30, _tmp29)
        tmp31 = tl.load(in_ptr6 + (r0_3 + 128*((((r0_3 + 128*x0 + 8192*x1) // 81920) % 32)) + 4096*(((x0 + 64*x1) % 640)) + 4956160*((r0_3 + 128*x0 + 8192*x1) // 2621440)), r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.load(in_ptr0 + (tl.broadcast_to(32*(((x0 + 64*x1) % 640)) + 38720*((x0 + 64*x1) // 20480) + ((((x0 + 64*x1) // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr0 + (tl.broadcast_to(19328 + 38720*(x2 // 20480) + (((x2 // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp6, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.where(tmp4, tmp33, tmp34)
        tmp36 = tl_math.exp(tmp35)
        tmp38 = tmp36 * tmp37
        tmp40 = tmp38 + tmp39
        tmp41 = tmp32 - tmp40
        tmp42 = tmp41 * tmp20
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(r0_mask, tmp45, _tmp44)
        tmp46 = ((63 + 64*x1) % 640)
        tmp47 = tmp46 >= tmp1
        tmp48 = tmp46 < tmp3
        tmp49 = tl.load(in_ptr0 + (tl.broadcast_to(19360 + 32*(((63 + 64*x1) % 640)) + 38720*((63 + 64*x1) // 20480) + ((((63 + 64*x1) // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp48, eviction_policy='evict_last', other=0.0)
        tmp50 = tmp46 >= tmp3
        tmp51 = tmp46 < tmp7
        tmp52 = tl.load(in_ptr0 + (tl.broadcast_to(38688 + 38720*((63 + 64*x1) // 20480) + ((((63 + 64*x1) // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp50, eviction_policy='evict_last', other=0.0)
        tmp53 = tl.where(tmp48, tmp49, tmp52)
        tmp54 = tl.load(in_ptr0 + (tl.broadcast_to(19360 + 32*(((x0 + 64*x1) % 640)) + 38720*((x0 + 64*x1) // 20480) + ((((x0 + 64*x1) // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr0 + (tl.broadcast_to(38688 + 38720*(x2 // 20480) + (((x2 // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp6, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.where(tmp4, tmp54, tmp55)
        tmp57 = tmp53 - tmp56
        tmp58 = tl_math.exp(tmp57)
        tmp60 = tmp58 * tmp59
        tmp62 = 0.08838834764831845
        tmp63 = tmp61 * tmp62
        tmp64 = tmp60 + tmp63
        tl.store(in_out_ptr0 + (r0_3 + 128*x2), tmp64, r0_mask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tl.store(out_ptr1 + (x2), tmp16, None)
    tl.store(out_ptr4 + (x2), tmp44, None)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp65 = tl.load(in_out_ptr0 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp66 = tl.load(in_ptr10 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.load(in_ptr11 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp87 = tl.load(in_ptr12 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp90 = tl.load(in_ptr13 + (r0_3 + 128*x2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp67 = tmp65 - tmp66
        tmp68 = (x2 % 640)
        tmp69 = tl.full([1, 1], 605, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tl.load(in_ptr3 + (tl.broadcast_to(19360 + 32*((x2 % 640)) + 38720*(x2 // 20480) + (((x2 // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp70, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp72 = tmp71.to(tl.float32)
        tmp74 = tmp73.to(tl.float32)
        tmp75 = tmp72 * tmp74
        tmp76 = tmp75 * tmp29
        tmp77 = tmp67 + tmp76
        tmp78 = tl.full([1, 1], 0, tl.int64)
        tmp79 = tmp68 >= tmp78
        tmp80 = tl.load(in_ptr0 + (tl.broadcast_to(32*(((x0 + 64*x1) % 640)) + 38720*((x0 + 64*x1) // 20480) + ((((x0 + 64*x1) // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp70, eviction_policy='evict_last', other=0.0)
        tmp81 = tmp68 >= tmp69
        tmp82 = tl.full([1, 1], 640, tl.int64)
        tmp83 = tmp68 < tmp82
        tmp84 = tl.load(in_ptr0 + (tl.broadcast_to(19328 + 38720*(x2 // 20480) + (((x2 // 640) % 32)), [XBLOCK, R0_BLOCK])), r0_mask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp85 = tl.where(tmp70, tmp80, tmp84)
        tmp86 = tl_math.exp(tmp85)
        tmp88 = tmp86 * tmp87
        tmp89 = tmp77 - tmp88
        tmp91 = tmp89 - tmp90
        tl.store(in_out_ptr0 + (r0_3 + 128*x2), tmp91, r0_mask)
    tmp92 = tl.load(in_out_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp93 = 0.08838834764831845
    tmp94 = tmp92 * tmp93
    tmp95 = tmp94 - tmp25
    tmp96 = (x2 % 640)
    tmp97 = tl.full([1, 1], 0, tl.int64)
    tmp98 = tmp96 >= tmp97
    tmp99 = tl.full([1, 1], 605, tl.int64)
    tmp100 = tmp96 < tmp99
    tmp101 = tl.load(in_ptr0 + (19360 + 32*(((x0 + 64*x1) % 640)) + 38720*((x0 + 64*x1) // 20480) + ((((x0 + 64*x1) // 640) % 32))), tmp100, eviction_policy='evict_last', other=0.0)
    tmp102 = tmp96 >= tmp99
    tmp103 = tl.full([1, 1], 640, tl.int64)
    tmp104 = tmp96 < tmp103
    tmp105 = tl.load(in_ptr0 + (38688 + 38720*(x2 // 20480) + (((x2 // 640) % 32))), tmp102, eviction_policy='evict_last', other=0.0)
    tmp106 = tl.where(tmp100, tmp101, tmp105)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2), tmp95, None)
    tl.store(out_ptr5 + (x2), tmp106, None)
