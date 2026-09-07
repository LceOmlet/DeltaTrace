

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 1}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_finite_decay_scan_0', 'backend_hash': 'F4337AF59E18A006A8110C1F352F0F4EB3D3AA586F5B0DB6F6A950B0D9B8F3BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'M': '*fp32', 'RawG0': '*fp32', 'G0': '*fp32', 'G1': '*fp32', 'Bterm': '*fp32', 'Dterm': '*fp32', 'Sterm': '*fp32', 'Out': '*fp32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'C': 64}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _finite_decay_scan(M, RawG0, G0, G1, Bterm, Dterm, Sterm, Out, C: tl.constexpr):
    """One [64,64] chunk/head; native state matrices never enter this scan."""
    block = tl.program_id(0)
    i = tl.arange(0, C)
    t = tl.arange(0, C)
    prev = tl.maximum(t - 1, 0)
    # Rows are future source i; columns are prefix cut t. Shift before scanning
    # so the inclusive affine scan yields the exclusive past P[t-1,i].
    m = tl.load(M + block*C*C + prev[None, :]*C + i[:, None], t[None, :] > 0, 0)
    a = tl.exp(tl.load(RawG0 + block*C + prev))
    a = tl.where(t > 0, a, 1.)
    a = tl.broadcast_to(a[None, :], (C, C))
    _, past = tl.associative_scan((a, m), 1, _affine_compose)
    g0i = tl.load(G0 + block*C + i)
    g0prev = tl.where(t > 0, tl.load(G0 + block*C + prev), 0.)
    g1i = tl.load(G1 + block*C + i)
    g1t = tl.load(G1 + block*C + t)
    e1 = tl.exp(tl.minimum(g1i[:, None] - g1t[None, :], 0.))
    e1 = tl.where(i[:, None] >= t[None, :], e1, 0.)
    ep = tl.exp(tl.minimum(g0prev[None, :] - g0i[:, None], 0.))
    ep = tl.where(i[:, None] < t[None, :], ep, 0.)
    b = tl.load(Bterm + block*C + i)
    d = tl.load(Dterm + block*C + i)
    s = tl.load(Sterm + block)
    end = tl.load(G1 + block*C + C - 1)
    out = (tl.exp(end - g1t) * (tl.exp(g0prev) * s + tl.sum(ep*d[:, None], 0))
           + tl.exp(g0prev) * tl.sum(e1*b[:, None], 0) + tl.sum(e1*past, 0))
    tl.store(Out + block*C + t, out)

@triton.jit
def _affine_compose(a_left, v_left, a_right, v_right):
    return a_right * a_left, v_right + a_right * v_left
