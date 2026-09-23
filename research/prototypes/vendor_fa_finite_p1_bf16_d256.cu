/* DeltaTrace finite P1 extension using the unmodified MetaX FA framework.
 * Framework portions Copyright (c) 2024, Tri Dao. BSD-3-Clause license retained.
 * Reuses Flash_fwd_kernel_traits, flash::copy, flash::gemm, flash::gemm_rs,
 * accumulator layouts, BF16 conversion, and FA's warp row reduction.
 * Framework source: MetaX-MACA/mcXformer aef88de756194e077460a1c8b8b341baa429259a.
 * This is an explicitly named finite-attribution operator, NOT model backward.
 * Q/K/V and U: BF16; MMA/softmax/row reductions: FP32; output: BF16.
 * Compact GQA inputs; query-head outputs retain the existing FP32 grouped reduction.
 * Endpoint means use the existing RHS shared tile; no global midpoint buffers.
 * D256; right-padding lengths; full padding tiles skip arithmetic and write zeros.
 * No integral quadrature, no global N-by-N buffer, no model/package patch.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"

using namespace cute;
#ifndef DELTATRACE_FINITE_TILE
#define DELTATRACE_FINITE_TILE 64
#endif
#ifndef DELTATRACE_FINITE_WARPS
#define DELTATRACE_FINITE_WARPS 4
#endif
using FiniteTraits=Flash_fwd_kernel_traits<256,DELTATRACE_FINITE_TILE,DELTATRACE_FINITE_TILE,
    DELTATRACE_FINITE_WARPS,false,false,mctlass::bfloat16_t>;

using CachedTraits=FiniteTraits;

struct FiniteParams {
    const void *q0,*k0,*q1,*k1,*v0,*u;
    const float *lse0,*lse1;
    float *tau,*center;
    void *dq,*dk,*dv;
    const int *valid_lengths;
    const int *coefficient_starts;
    int batch,heads,kv_heads,length,query_start;
    float scale;
};

// Phase 0: row normalization and finite center.
// Phase 1: query-owned Q multiplier. Phase 2: key-owned K/V multipliers.
template<int Phase,typename Traits=FiniteTraits>
__global__ void deltatrace_fa_finite_p1_kernel(FiniteParams p) {
    using E=typename Traits::Element;
    // Like native Flash_fwd_params: opaque storage is typed inside each
    // compilation pass. The vendor traits use half_t on the host pass and
    // BF16 on the actual MACA device pass; do not override the vendor header.
    const E *q0=static_cast<const E*>(p.q0), *k0=static_cast<const E*>(p.k0);
    const E *q1=static_cast<const E*>(p.q1), *k1=static_cast<const E*>(p.k1);
    const E *v0=static_cast<const E*>(p.v0), *u=static_cast<const E*>(p.u);
    E *dq=static_cast<E*>(p.dq), *dk=static_cast<E*>(p.dk), *dv=static_cast<E*>(p.dv);
    constexpr int M=Traits::kBlockM,N=Traits::kBlockN,D=Traits::kHeadDim;
    const int tid=threadIdx.x, bh=blockIdx.y, row0=blockIdx.x*M+p.query_start;
    // Padded length remains the storage stride; valid bounds are per sample.
    const int length=p.length, valid=p.valid_lengths[bh/p.heads];
    const int start=p.coefficient_starts?p.coefficient_starts[bh/p.heads]:0;
    const int query_length=length-p.query_start;
    // Q/U and requested coefficients contain only the cached suffix. K/V
    // retain the complete native history. Keep global causal coordinates.
    const int64_t head_offset=(int64_t(bh)*query_length-p.query_start)*D;
    // Same GQA mapping as pinned FA flash_fwd_kernel.h: bidh / h_h_k_ratio.
    const int kv_bh=(bh/p.heads)*p.kv_heads+(bh%p.heads)/(p.heads/p.kv_heads);
    const int64_t kv_offset=int64_t(kv_bh)*length*D;
    const int64_t pair_left_offset=Phase==2?kv_offset:head_offset;
    const int64_t pair_right_offset=Phase==2?head_offset:kv_offset;
    // Every padding output is written, including entire empty tail tiles.
    // This branch is uniform within the block, before any synchronization.
    // Causal Q rows depend only on their own upstream row. K/V rows at or
    // after start depend only on query rows at or after those K/V rows.
    // Thus the requested suffix coefficients use the original full key
    // history, while earlier coefficient output tiles can be omitted.
    if(row0>=valid || row0+M<=start) {
        if constexpr(Phase==0) {
            for(int j=tid;j<M;j+=blockDim.x) if(row0+j<length) {
                const int64_t pos=int64_t(bh)*query_length+row0+j-p.query_start;
                p.tau[pos]=0.f;p.center[pos]=0.f;
            }
        } else {
            E *out=Phase==1?dq:dk;
            for(int j=tid;j<M*D;j+=blockDim.x) if(row0+j/D<length) {
                const int64_t pos=head_offset+int64_t(row0)*D+j;
                out[pos]=E(0.f);
                if constexpr(Phase==2) dv[pos]=E(0.f);
            }
        }
        return;
    }
    extern __shared__ char scratch[];
    auto sA=make_tensor(make_smem_ptr(reinterpret_cast<E*>(scratch)),typename Traits::SmemLayoutQ{});
    auto sB=make_tensor(sA.data()+size(sA),typename Traits::SmemLayoutKV{});
    auto sBt=make_tensor(sB.data(),typename Traits::SmemLayoutVtransposed{});
    auto sBtPlain=make_tensor(sB.data(),typename Traits::SmemLayoutVtransposedNoSwizzle{});
    // Form the midpoint only after score products finish. Retaining a whole
    // RHS tile across those GEMMs caused register spills on the MetaX path.

    typename Traits::GmemTiledCopyQKV global_copy;
    auto global_thread=global_copy.get_thread_slice(tid);
    auto toA=global_thread.partition_D(sA);
    auto toB=global_thread.partition_D(sB);
    auto coordA=make_identity_tensor(Shape<Int<M>,Int<D>>{});
    auto coordB=make_identity_tensor(Shape<Int<N>,Int<D>>{});
    auto coordsA=global_thread.partition_S(coordA);
    auto coordsB=global_thread.partition_S(coordB);
    auto predA=make_tensor<bool>(make_shape(size<2>(toA)));
    auto predB=make_tensor<bool>(make_shape(size<2>(toB)));

    typename Traits::TiledMma mma;
    auto mma_thread=mma.get_thread_slice(tid);
    auto regA=mma_thread.partition_fragment_A(sA);
    auto regB=mma_thread.partition_fragment_B(sB);
    auto regBt=mma_thread.partition_fragment_B(sBtPlain);
    auto copyA=make_tiled_copy_A(typename Traits::SmemCopyAtom{},mma);
    auto copyB=make_tiled_copy_B(typename Traits::SmemCopyAtom{},mma);
    auto threadA=copyA.get_thread_slice(tid);
    auto threadB=copyB.get_thread_slice(tid);
    auto fromA=threadA.partition_S(sA);
    auto fromB=threadB.partition_S(sB);
    auto copyBt=make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{},mma);
    auto threadBt=copyBt.get_thread_slice(tid);
    auto fromBt=threadBt.partition_S(sBt);

    auto coords=mma_thread.partition_C(make_identity_tensor(Shape<Int<M>,Int<N>>{}));
    auto rowcoords=logical_divide(coords,Shape<_4>{})(make_coord(0,_),_,0);
    auto denominator=make_tensor<float>(Shape<Int<decltype(size(rowcoords))::value>>{});
    auto numerator=make_fragment_like(denominator);
    clear(denominator);clear(numerator);
    auto accQ=partition_fragment_C(mma,Shape<Int<M>,Int<D>>{});
    auto accV=make_fragment_like(accQ);
    clear(accQ);clear(accV);

    // Reuse the historical Qwen3 register-retention schedule through the
    // same vendor A_in_regs GEMM and the unchanged D256 tile layout.
    auto owner0=make_fragment_like(regA);
    auto owner1=make_fragment_like(regA);
    auto ownerU=make_fragment_like(regA);
    if constexpr(Phase!=0) {
        auto load_owner=[&](const E *left,auto &reg) {
            __syncthreads();
            auto gA=make_tensor(make_gmem_ptr(left+pair_left_offset+int64_t(row0)*D),
                               Shape<Int<M>,Int<D>>{},Stride<Int<D>,_1>{});
            auto fromGA=global_thread.partition_S(gA);
            flash::copy<false,true,true>(global_copy,fromGA,toA,coordsA,predA,valid-row0);
            __syncthreads();
            auto register_view=threadA.retile_D(reg);
            cute::copy(copyA,fromA,register_view);
            __syncthreads();
        };
        if constexpr(Phase==2) {load_owner(k0,owner0);load_owner(k1,owner1);load_owner(v0,ownerU);}
        else {load_owner(q0,owner0);load_owner(q1,owner1);load_owner(u,ownerU);}
    }
    const int begin=Phase==2 ? row0 : 0;
    const int end=Phase==2 ? valid : min(valid,row0+M);
    for(int col0=begin;col0<end;col0+=N) {
        auto pair=[&](const E *left,const E *right,auto &acc,int endpoint) __attribute__((always_inline)) {
            __syncthreads();
            auto gA=make_tensor(make_gmem_ptr(left+pair_left_offset+int64_t(row0)*D),
                               Shape<Int<M>,Int<D>>{},Stride<Int<D>,_1>{});
            auto gB=make_tensor(make_gmem_ptr(right+pair_right_offset+int64_t(col0)*D),
                               Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
            auto fromGA=global_thread.partition_S(gA);
            auto fromGB=global_thread.partition_S(gB);
            if constexpr(Phase==0) flash::copy<false,true,true>(global_copy,fromGA,toA,coordsA,predA,valid-row0);
            flash::copy<false,true,true>(global_copy,fromGB,toB,coordsB,predB,valid-col0);
            __syncthreads();
            clear(acc);
            if constexpr(Phase==0) flash::gemm_opt(acc,regA,regB,fromA,fromB,mma,copyA,copyB,threadA,threadB);
            else if(endpoint==0) flash::gemm_opt<true,false>(acc,owner0,regB,fromA,fromB,mma,copyA,copyB,threadA,threadB);
            else if(endpoint==1) flash::gemm_opt<true,false>(acc,owner1,regB,fromA,fromB,mma,copyA,copyB,threadA,threadB);
            else flash::gemm_opt<true,false>(acc,ownerU,regB,fromA,fromB,mma,copyA,copyB,threadA,threadB);
        };
        auto a0=partition_fragment_C(mma,Shape<Int<M>,Int<N>>{});
        auto a1=make_fragment_like(a0);
        auto at=make_fragment_like(a0);
        if constexpr(Phase==2) {
            pair(k0,q0,a0,0);pair(k1,q1,a1,1);pair(v0,u,at,2);
        } else {
            pair(q0,k0,a0,0);pair(q1,k1,a1,1);pair(u,v0,at,2);
        }
        #pragma unroll
        for(int i=0;i<size(a0);++i) {
            const int r=row0+get<0>(coords(i)),c=col0+get<1>(coords(i));
            const int query=Phase==2?c:r,key=Phase==2?r:c;
            float lm=0.f,actual_p1=0.f;
            if(query<valid && key<valid && key<=query) {
                const int64_t pos=int64_t(bh)*query_length+query-p.query_start;
                const float lp0=a0(i)*p.scale-p.lse0[pos];
                const float lp1=a1(i)*p.scale-p.lse1[pos];
                const float distance=fabsf(lp1-lp0);
                const float ratio=distance==0.f?1.f:-expm1f(-distance)/distance;
                lm=__builtin_exp2f(fmaxf(lp0,lp1)*float(M_LOG2E))*ratio;
                actual_p1=__builtin_exp2f(lp1*float(M_LOG2E));
                if constexpr(Phase!=0) a0(i)=lm*(at(i)-p.center[pos]);
            } else if constexpr(Phase!=0) a0(i)=0.f;
            if constexpr(Phase==0) {a0(i)=lm;a1(i)=lm*at(i);}
            else a1(i)=actual_p1;
        }
        if constexpr(Phase==0) {
            auto lrows=make_tensor(a0.data(),flash::convert_layout_acc_rowcol(a0.layout()));
            auto trows=make_tensor(a1.data(),flash::convert_layout_acc_rowcol(a1.layout()));
            flash::SumOp<float> sum;
            flash::thread_reduce_<false>(lrows,denominator,sum);
            flash::thread_reduce_<false>(trows,numerator,sum);
        } else {
            auto multiply=[&](auto &weights,const E *values,auto &acc,bool shared_mean) __attribute__((always_inline)) {
                __syncthreads();
                if(shared_mean) {
                    const E *right0=Phase==1?k0:q0;
                    const E *right1=Phase==1?k1:q1;
                    auto gB0=make_tensor(make_gmem_ptr(right0+pair_right_offset+int64_t(col0)*D),
                                        Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
                    auto gB1=make_tensor(make_gmem_ptr(right1+pair_right_offset+int64_t(col0)*D),
                                        Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
                    auto fromGB0=global_thread.partition_S(gB0);
                    auto fromGB1=global_thread.partition_S(gB1);
                    flash::copy<false,true,true>(global_copy,fromGB0,toB,coordsB,predB,valid-col0);
                    __syncthreads();
                    auto firstEndpointB=make_fragment_like(toB);
                    cute::copy(toB,firstEndpointB);
                    __syncthreads();
                    flash::copy<false,true,true>(global_copy,fromGB1,toB,coordsB,predB,valid-col0);
                    __syncthreads();
                    #pragma unroll
                    for(int j=0;j<size(firstEndpointB);++j)
                        toB(j)=E((float(firstEndpointB(j))+float(toB(j)))*0.5f);
                } else {
                auto gB=make_tensor(make_gmem_ptr(values+(Phase==1?kv_offset:head_offset)+int64_t(col0)*D),
                                   Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
                auto fromGB=global_thread.partition_S(gB);
                flash::copy<false,true,true>(global_copy,fromGB,toB,coordsB,predB,valid-col0);
                }
                __syncthreads();
                CONVERT_TENSOR_TYPE(float,E,weights,half_weights)
                auto regWeights=make_tensor(half_weights.data(),weights.layout());
                // The vendor copy_trans fast path consumes a 128-column
                // operand (four source pointer pairs). Invoke that owner
                // helper on two views of D256, rather than indexing beyond
                // its native source pointer array.
                uint32_t offsets[5];
                const uint32_t lane=__lane_id();
                const uint32_t rr=((lane>>4)<<2)+(lane&3);
                const uint32_t cc=((lane>>2)&3)<<1;
                #pragma unroll
                for(int j=0;j<4;++j) offsets[j]=((((cc+j*8)>>2)^(rr&7))<<2)+(cc&3)+rr*32;
                offsets[4]=((lane&7)<<1)+((lane>>5)<<10);
                #pragma unroll
                for(int part=0;part<2;++part) {
                    auto sharedHalf=local_tile(sBt,Shape<_128,Int<N>>{},make_coord(part,0));
                    auto plainHalf=local_tile(sBtPlain,Shape<_128,Int<N>>{},make_coord(part,0));
                    auto registerHalf=mma_thread.partition_fragment_B(plainHalf);
                    auto fromHalf=threadBt.partition_S(sharedHalf);
                    auto accHalf=make_tensor(acc.data()+part*8*get<2>(stride(acc)),
                        make_layout(make_shape(get<0>(shape(acc)),get<1>(shape(acc)),_8{}),stride(acc)));
                    const uint32_t sourceStride=get<1>(get<1>(fromHalf(_,_,_0{}).layout().layout_fn().stride()))/2;
                    const uint32_t destinationStride=get<1>(get<1>(registerHalf(_,_,_0{}).layout().stride()))/2;
                    flash::gemm_rs<false,true>(accHalf,regWeights,registerHalf,fromHalf,mma,offsets,sourceStride,destinationStride);
                }
            };
            if constexpr(Phase==1) multiply(a0,nullptr,accQ,true);
            else {multiply(a0,nullptr,accQ,true);multiply(a1,u,accV,false);}
        }
    }
    if constexpr(Phase==0) {
        flash::quadreduce_sum(denominator);flash::quadreduce_sum(numerator);
        if(get<1>(rowcoords(0))==0) {
            #pragma unroll
            for(int i=0;i<size(denominator);++i) {
                const int r=row0+get<0>(rowcoords(i));
                if(r<length) {
                    p.tau[int64_t(bh)*query_length+r-p.query_start]=r<valid?denominator(i):0.f;
                    p.center[int64_t(bh)*query_length+r-p.query_start]=r<valid?numerator(i)/denominator(i):0.f;
                }
            }
        }
    } else {
        auto outcoords=mma_thread.partition_C(make_identity_tensor(Shape<Int<M>,Int<D>>{}));
        E *out=Phase==1?dq:dk;
        #pragma unroll
        for(int i=0;i<size(accQ);++i) {
            const int r=row0+get<0>(outcoords(i)),d=get<1>(outcoords(i));
            if(r<length) {
                const int64_t pos=head_offset+int64_t(r)*D+d;
                out[pos]=E(r<valid && r>=start?accQ(i)*p.scale:0.f);
                if constexpr(Phase==2) dv[pos]=E(r<valid && r>=start?accV(i):0.f);
            }
        }
    }
}

static int launch_finite(FiniteParams p,void *stream_ptr) {
    using E=mctlass::bfloat16_t;
    if(p.batch<1||p.heads<1||p.kv_heads<1||p.heads%p.kv_heads!=0||p.length<1)return -1;
    if(p.query_start<0||p.query_start>=p.length||p.query_start%FiniteTraits::kBlockM)return -2;
    dim3 grid((p.length-p.query_start+FiniteTraits::kBlockM-1)/FiniteTraits::kBlockM,p.batch*p.heads);
    constexpr int shared=(size(typename FiniteTraits::SmemLayoutQ{})+size(typename FiniteTraits::SmemLayoutKV{}))*sizeof(E);
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    deltatrace_fa_finite_p1_kernel<0><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    dim3 cachedGrid((p.length-p.query_start+CachedTraits::kBlockM-1)/CachedTraits::kBlockM,p.batch*p.heads);
    constexpr int cachedShared=(size(typename CachedTraits::SmemLayoutQ{})+size(typename CachedTraits::SmemLayoutKV{}))*sizeof(E);
    deltatrace_fa_finite_p1_kernel<1,CachedTraits><<<cachedGrid,CachedTraits::kNThreads,cachedShared,stream>>>(p);
    error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    deltatrace_fa_finite_p1_kernel<2,CachedTraits><<<cachedGrid,CachedTraits::kNThreads,cachedShared,stream>>>(p);
    return int(cudaGetLastError());
}

extern "C" int deltatrace_fa_finite_p1_bf16_d256(
    const void *q0,const void *k0,const void *q1,const void *k1,const void *v0,
    const void *u,const void *lse0,const void *lse1,
    void *tau,void *center,void *dq,void *dk,void *dv,const void *valid_lengths,int batch,int heads,int kv_heads,int length,
    float scale,void *stream_ptr) {
    FiniteParams p{q0,k0,q1,k1,v0,u,(const float*)lse0,(const float*)lse1,
        (float*)tau,(float*)center,dq,dk,dv,(const int*)valid_lengths,nullptr,batch,heads,kv_heads,length,0,scale};
    return launch_finite(p,stream_ptr);
}

extern "C" int deltatrace_fa_finite_p1_bf16_d256_suffix(
    const void *q0,const void *k0,const void *q1,const void *k1,const void *v0,
    const void *u,const void *lse0,const void *lse1,
    void *tau,void *center,void *dq,void *dk,void *dv,const void *valid_lengths,
    const void *coefficient_starts,int batch,int heads,int kv_heads,int length,
    float scale,void *stream_ptr) {
    FiniteParams p{q0,k0,q1,k1,v0,u,(const float*)lse0,(const float*)lse1,
        (float*)tau,(float*)center,dq,dk,dv,(const int*)valid_lengths,
        (const int*)coefficient_starts,batch,heads,kv_heads,length,0,scale};
    return launch_finite(p,stream_ptr);
}

extern "C" int deltatrace_fa_finite_p1_bf16_d256_cached_suffix(
    const void *q0,const void *k0,const void *q1,const void *k1,const void *v0,
    const void *u,const void *lse0,const void *lse1,
    void *tau,void *center,void *dq,void *dk,void *dv,const void *valid_lengths,
    const void *coefficient_starts,int batch,int heads,int kv_heads,int length,int query_start,
    float scale,void *stream_ptr) {
    FiniteParams p{q0,k0,q1,k1,v0,u,(const float*)lse0,(const float*)lse1,
        (float*)tau,(float*)center,dq,dk,dv,(const int*)valid_lengths,
        (const int*)coefficient_starts,batch,heads,kv_heads,length,query_start,scale};
    return launch_finite(p,stream_ptr);
}
