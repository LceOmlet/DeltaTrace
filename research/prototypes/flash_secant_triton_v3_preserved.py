"""Isolated fused finite-secant attention pullback; NOT model autograd or forward.

Tile scheduling follows the key-block backward strategy in the installed
FlashAttention Triton implementation (Dao-AILab/flash-attention, BSD-3-Clause).
The derivative equations are deliberately replaced by the two-endpoint rule.
No native package is patched, no forward is implemented, and no torch.autograd
function is registered. Inputs are actual native FA Q/K/V and natural-log LSE.

FP32 upstream/means/products remain FP32, with IEEE tl.dot for these products.
QK products use original FP16 operands with FP32 accumulation. dQ uses FP32
atomic accumulation as in nondeterministic FlashAttention backward; differences
must be measured. FP64 streaming audit credits are returned, never discarded.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

COMPILED_KERNELS={}

@triton.jit
def _logmean(lp0,lp1,valid):
    distance=tl.where(valid,tl.abs(lp1-lp0),0.)
    denom=tl.where(distance>0.,distance,1.)
    ratio=tl.where(distance>0.,-libdevice.expm1(-distance)/denom,1.)
    return tl.where(valid,tl.exp(tl.maximum(lp0,lp1))*ratio,0.)

@triton.jit
def _secant_center(Q0,K0,V0,LSE0,Q1,K1,V1,LSE1,U,C,TAU,AUDIT_PV,
                   N:tl.constexpr,H:tl.constexpr,KV:tl.constexpr,D:tl.constexpr,
                   SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    qm=tl.program_id(0)*BM;bh=tl.program_id(1);b=bh//H;head=bh%H;kh=head//(H//KV)
    qi=qm+tl.arange(0,BM);ki=tl.arange(0,BN);di=tl.arange(0,D)
    qp=(b*N*H+qi[:,None]*H+head)*D+di[None,:]
    q0=tl.load(Q0+qp,mask=qi[:,None]<N,other=0.)
    q1=tl.load(Q1+qp,mask=qi[:,None]<N,other=0.)
    up=(bh*N+qi[:,None])*D+di[None,:]
    u=tl.load(U+up,mask=qi[:,None]<N,other=0.)
    l0=tl.load(LSE0+bh*N+qi,mask=qi<N,other=0.)
    l1=tl.load(LSE1+bh*N+qi,mask=qi<N,other=0.)
    tau=tl.full((BM,),0.,tl.float32);numerator=tl.full((BM,),0.,tl.float32)
    audit=tl.full((BM,),0.,tl.float64)
    for start in range(0,tl.minimum(qm+BM,N),BN):
        keys=start+ki;kp=(b*N*KV+keys[:,None]*KV+kh)*D+di[None,:]
        k0=tl.load(K0+kp,mask=keys[:,None]<N,other=0.)
        k1=tl.load(K1+kp,mask=keys[:,None]<N,other=0.)
        v0=tl.load(V0+kp,mask=keys[:,None]<N,other=0.).to(tl.float32)
        v1=tl.load(V1+kp,mask=keys[:,None]<N,other=0.).to(tl.float32)
        s0=tl.dot(q0,tl.trans(k0))*SCALE;s1=tl.dot(q1,tl.trans(k1))*SCALE
        valid=(qi[:,None]<N)&(keys[None,:]<N)&(keys[None,:]<=qi[:,None])
        lp0=tl.where(valid,s0-l0[:,None],float('-inf'));lp1=tl.where(valid,s1-l1[:,None],float('-inf'))
        mean=_logmean(lp0,lp1,valid)
        dp=tl.dot(u,tl.trans((v0+v1)*.5),input_precision='ieee')
        tau+=tl.sum(mean,1);numerator+=tl.sum(mean*dp,1)
        delta_p=tl.exp(lp1)-tl.exp(lp0)
        audit+=tl.sum(dp.to(tl.float64)*delta_p.to(tl.float64),1)
    center=tl.where(tau>0.,numerator/tau,float('nan'))
    tl.store(C+bh*N+qi,center,mask=qi<N);tl.store(TAU+bh*N+qi,tau,mask=qi<N)
    tl.store(AUDIT_PV+bh*N+qi,audit,mask=qi<N)

@triton.jit
def _secant_backward_keyblock(Q0,K0,V0,LSE0,Q1,K1,V1,LSE1,U,C,DQ,DKH,DVH,AUDIT_SOFT,
                             N:tl.constexpr,H:tl.constexpr,KV:tl.constexpr,D:tl.constexpr,
                             SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    block=tl.program_id(0);bh=tl.program_id(1);b=bh//H;head=bh%H;kh=head//(H//KV)
    keys=block*BN+tl.arange(0,BN);di=tl.arange(0,D)
    kp=(b*N*KV+keys[:,None]*KV+kh)*D+di[None,:]
    k0=tl.load(K0+kp,mask=keys[:,None]<N,other=0.)
    k1=tl.load(K1+kp,mask=keys[:,None]<N,other=0.)
    v0=tl.load(V0+kp,mask=keys[:,None]<N,other=0.).to(tl.float32)
    v1=tl.load(V1+kp,mask=keys[:,None]<N,other=0.).to(tl.float32)
    kbar=(k0.to(tl.float32)+k1.to(tl.float32))*.5;vbar=(v0+v1)*.5
    dk=tl.full((BN,D),0.,tl.float32);dv=tl.full((BN,D),0.,tl.float32)
    audit=tl.full((),0.,tl.float64)
    for start in range(block*BN,N,BM):
        qi=start+tl.arange(0,BM);qp=(b*N*H+qi[:,None]*H+head)*D+di[None,:]
        q0=tl.load(Q0+qp,mask=qi[:,None]<N,other=0.)
        q1=tl.load(Q1+qp,mask=qi[:,None]<N,other=0.)
        up=(bh*N+qi[:,None])*D+di[None,:]
        u=tl.load(U+up,mask=qi[:,None]<N,other=0.)
        l0=tl.load(LSE0+bh*N+qi,mask=qi<N,other=0.);l1=tl.load(LSE1+bh*N+qi,mask=qi<N,other=0.)
        center=tl.load(C+bh*N+qi,mask=qi<N,other=0.)
        s0=tl.dot(q0,tl.trans(k0))*SCALE;s1=tl.dot(q1,tl.trans(k1))*SCALE
        valid=(qi[:,None]<N)&(keys[None,:]<N)&(keys[None,:]<=qi[:,None])
        lp0=tl.where(valid,s0-l0[:,None],float('-inf'));lp1=tl.where(valid,s1-l1[:,None],float('-inf'))
        mean=_logmean(lp0,lp1,valid);pbar=(tl.exp(lp0)+tl.exp(lp1))*.5
        dp=tl.dot(u,tl.trans(vbar),input_precision='ieee')
        ds=mean*(dp-center[:,None])
        dq=tl.dot(ds,kbar,input_precision='ieee')*SCALE
        tl.atomic_add(DQ+up,dq,mask=qi[:,None]<N,sem='relaxed')
        qbar=(q0.to(tl.float32)+q1.to(tl.float32))*.5
        dk+=tl.dot(tl.trans(ds),qbar,input_precision='ieee')
        dv+=tl.dot(tl.trans(pbar),u,input_precision='ieee')
        delta_score=tl.where(valid,s1-s0,0.)
        audit+=tl.sum(tl.sum(ds.to(tl.float64)*delta_score.to(tl.float64),0),0)
    out=(bh*N+keys[:,None])*D+di[None,:]
    tl.store(DKH+out,dk*SCALE,mask=keys[:,None]<N);tl.store(DVH+out,dv,mask=keys[:,None]<N)
    tl.store(AUDIT_SOFT+bh*tl.cdiv(N,BN)+block,audit)

def finite_attention_pullback(q0,k0,v0,lse0,q1,k1,v1,lse1,upstream):
    """All Q/K/V are actual FA [B,N,H,D], upstream is FP32 [B,H,N,D]."""
    assert q0.shape==q1.shape and k0.shape==k1.shape==v0.shape==v1.shape
    b,n,h,d=q0.shape;kv=k0.shape[2]
    assert k0.shape[:2]==(b,n) and h%kv==0 and d==128
    assert all(x.dtype==torch.float16 and x.is_cuda for x in [q0,k0,v0,q1,k1,v1])
    # MetaX host-to-device copies can select noncontiguous device layouts.
    # Pack explicitly inside the charged operator, never reinterpret strides.
    q0,k0,v0,q1,k1,v1=[x.contiguous() for x in [q0,k0,v0,q1,k1,v1]]
    assert all(x.is_contiguous() for x in [q0,k0,v0,q1,k1,v1])
    assert upstream.shape==(b,h,n,d) and upstream.dtype==torch.float32
    assert lse0.shape==lse1.shape==(b,h,n) and lse0.dtype==lse1.dtype==torch.float32
    u=upstream.contiguous();l0=lse0.contiguous();l1=lse1.contiguous()
    center=torch.empty((b,h,n),device=q0.device,dtype=torch.float32);tau=torch.empty_like(center)
    audit_pv=torch.empty((b,h,n),device=q0.device,dtype=torch.float64)
    dq=torch.zeros((b,h,n,d),device=q0.device,dtype=torch.float32)
    dkh=torch.empty_like(dq);dvh=torch.empty_like(dq)
    bm,bn=16,16
    audit_soft=torch.empty((b,h,triton.cdiv(n,bn)),device=q0.device,dtype=torch.float64)
    args=(q0,k0,v0,l0,q1,k1,v1,l1,u)
    COMPILED_KERNELS['center']=_secant_center[(triton.cdiv(n,bm),b*h)](*args,center,tau,audit_pv,n,h,kv,d,d**-.5,bm,bn,num_warps=4,num_stages=1)
    COMPILED_KERNELS['backward']=_secant_backward_keyblock[(triton.cdiv(n,bn),b*h)](*args,center,dq,dkh,dvh,audit_soft,n,h,kv,d,d**-.5,bm,bn,num_warps=4,num_stages=1)
    return {'dq':dq,'dk_per_query_head':dkh,'dv_per_query_head':dvh,
        'dk':dkh.reshape(b,kv,h//kv,n,d).sum(2),'dv':dvh.reshape(b,kv,h//kv,n,d).sum(2),
        'center':center,'logmean_row_mass':tau,'pv_probability_credit':audit_pv.sum(),'softmax_score_credit':audit_soft.sum(),
        'scope':'Isolated finite attribution pullback, not native model gradient. Actual paired QKV/LSE. FP32 atomics for dQ; FP64 streaming audit. No dense NxN device tensor.'}
