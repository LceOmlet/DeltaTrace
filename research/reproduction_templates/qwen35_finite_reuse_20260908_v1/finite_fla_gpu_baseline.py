"""GPU mixed finite FLA pullback on actual native endpoint intermediates.

Native FLA supplies reverse chunk-state propagation; existing torch.bmm supplies
BF16 operand / FP32 output GEMMs. The only new Triton operation is the finite
decay contraction's affine prefix scan. No model forward, attention probability
or substitute ordinary backward is implemented here. This initial GPU version
is not yet a fused production backend or complete model attribution method.
"""
import hashlib
import importlib
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _affine_compose(a_left, v_left, a_right, v_right):
    return a_right * a_left, v_right + a_right * v_left


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


def verify_native_sources(expected_sources):
    chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    state = importlib.import_module('fla.ops.common.chunk_delta_h')
    wy = importlib.import_module('fla.ops.gated_delta_rule.wy_fast')
    for name, module in {'chunk':chunk, 'state':state, 'wy':wy}.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected_sources[name]:
            raise ValueError('Unreviewed native FLA source: ' + name)
    assert chunk.chunk_gated_delta_rule_bwd_dhu is state.chunk_gated_delta_rule_bwd_dhu
    return chunk


def native_input_adjoints(endpoints, do, scale):
    """Unchanged FLA stages, already normalized operands, zero terminal adjoint."""
    chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    q, k, w, g = [endpoints[name][1::2].contiguous() for name in ['q','k','w','g']]
    do = do.contiguous()
    local = chunk.chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=scale, cu_seqlens=None)
    dh, dh0, du = chunk.chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, g=g, do=do,
        dv=local, h0=None, dht=None, scale=scale, cu_seqlens=None)
    assert dh0 is None
    return {'do':do, 'dh_end':dh, 'dU_WY':du}


def _mm(a, b):
    # Reuse the installed vendor GEMM, including its normal BF16 accumulation
    # path. No custom matrix multiplication kernel or elevated input precision.
    return torch.bmm(a.to(torch.bfloat16), b.to(torch.bfloat16), out_dtype=torch.float32)


def _T(x):
    return x.transpose(-1, -2)


def _dot(x, y):
    return (x*y).sum(-1)


def mixed_coefficients(endpoints, adjoints, scale):
    """All samples/heads/chunks batched; inputs already on the same GPU.

    Interleaved EOS/input endpoints, fixed prefix length per row. Right-padding
    here is only for the final64-token tile, not a claim of variable-length model
    mask support. Source identity must be checked once before invoking this API.
    """
    E, T, H, K = endpoints['q'].shape
    assert E % 2 == 0 and K == 128 and endpoints['v'].shape == endpoints['q'].shape
    assert endpoints['q'].is_cuda and endpoints['q'].dtype == torch.bfloat16
    B, N, C = E//2, triton.cdiv(T,64), 64
    count = B*H*N
    def pack_tensor(x, cumulative=False):
        # [B,T,H,D?] -> [B*H*N,64,D?]. Padding is an identity state transition:
        # zero q/k/v/beta/do and raw g; cumulative g carries its last real value.
        assert x.shape[:3] == (B,T,H)
        vector = x.ndim == 4
        if vector:
            a=x.permute(0,2,1,3)
            a=F.pad(a,(0,0,0,N*C-T))
            return a.reshape(count,C,x.shape[-1]).contiguous()
        a=x.permute(0,2,1)
        if cumulative and N*C>T:
            a=torch.cat((a,a[...,-1:].expand(B,H,N*C-T)),dim=-1)
        else:
            a=F.pad(a,(0,N*C-T))
        return a.reshape(count,C).contiguous()
    def token(name, endpoint):
        return pack_tensor(endpoints[name][endpoint::2], cumulative=name=='g')
    q1,k0,k1,v0,u0=[token(name,ep) for name,ep in [('q',1),('k',0),('k',1),('v',0),('v_new',0)]]
    g0,G0,G1,beta1=[token(name,ep).float() for name,ep in [('raw_g',0),('g',0),('g',1),('beta',1)]]
    A1=token('A',1)
    Z=pack_tensor(adjoints['do'])
    L=_mm(_T(A1),pack_tensor(adjoints['dU_WY']))
    W=beta1[...,None]*L
    H0=endpoints['h'][0::2].permute(0,2,1,3,4).reshape(count,K,K).contiguous()
    D=adjoints['dh_end'].permute(0,2,1,3,4).reshape(count,K,K).contiguous()
    row=torch.arange(C,device=q1.device)
    causal=row[:,None]>=row[None,:];lower=row[:,None]>row[None,:];upper=causal.T
    E0=torch.exp((G0[:,:,None]-G0[:,None,:]).clamp_max(0))*causal
    E1=torch.exp((G1[:,None,:]-G1[:,:,None]).clamp_max(0))*upper
    eend=torch.exp(G1[:,-1:]-G1)
    eg0=torch.exp(G0)
    # Share transposed pair products rather than recomputing them.
    UZ=_mm(u0,_T(Z)); UW=_mm(u0,_T(W))
    K0Q=_mm(k0,_T(q1)); K0K1=_mm(k0,_T(k1))
    K0H=_mm(k0,H0)
    dq=scale*(eg0[...,None]*_mm(Z,_T(H0))+_mm(_T(UZ)*E0,k0))
    dk=(eend[...,None]*_mm(u0,_T(D)) + scale*_mm(UZ*E1,q1) - _mm(UW*E1,k1)
        + beta1[...,None]*k1.float()*_dot(u0.float(),L)[...,None]
        - eg0[...,None]*_mm(W,_T(H0)) - _mm(_T(UW)*(E0*lower),k0))
    r0=eg0[...,None]*K0H+_mm(_mm(k0,_T(k0))*(E0*lower),u0)
    dbeta=_dot(v0.float()-r0,L)
    S=(H0.float()*D.float()).sum((-1,-2))
    b=scale*_dot(Z.float(),_mm(q1,H0))-_dot(W,_mm(k1,H0))
    d=_dot(u0.float(),_mm(k0,D))
    M=scale*K0Q*UZ-K0K1*UW
    dalpha=torch.empty_like(g0)
    _finite_decay_scan[(count,)](M,g0,G0,G1,b,d,S,dalpha,C=C,num_warps=4,num_stages=1)
    raw1=token('raw_g',1).float();distance=(raw1-g0).abs()
    denominator=torch.where(distance==0,torch.ones_like(distance),distance)
    ratio=torch.where(distance==0,torch.ones_like(distance),-torch.expm1(-distance)/denominator)
    dg=dalpha*torch.exp(torch.maximum(g0,raw1))*ratio
    def unpack(x):
        if x.ndim==3:
            return x.reshape(B,H,N*C,K)[:,:,:T].permute(0,2,1,3).contiguous()
        return x.reshape(B,H,N*C)[:,:,:T].permute(0,2,1).contiguous()
    return {key:unpack(value) for key,value in {'q':dq,'k':dk,'v':W,'beta':dbeta,'alpha':dalpha,'g':dg}.items()}


def finite_fla_pullback(endpoints, do, scale):
    """Full local mixed pullback, including packing and two native adjoint stages."""
    return mixed_coefficients(endpoints,native_input_adjoints(endpoints,do,scale),scale)


def make_compiled_finite_pullback():
    """Use the official compiler to fuse layout/scalar work; keep native FLA calls.

    Compile time and steady cost must be recorded separately. CUDA graphs and
    autotune searches are disabled; this wrapper does not cache model endpoints
    or computed coefficients. Source identity must be verified by its caller.
    """
    compiled = torch.compile(mixed_coefficients, fullgraph=True, dynamic=False,
        options={'triton.cudagraphs':False, 'max_autotune':False})
    def pullback(endpoints, do, scale):
        return compiled(endpoints, native_input_adjoints(endpoints,do,scale), scale)
    return pullback
