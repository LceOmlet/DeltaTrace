/* DeltaTrace finite P1 extension using the unmodified MetaX FA framework.
 * Framework portions Copyright (c) 2024, Tri Dao. BSD-3-Clause license retained.
 * Reuses Flash_fwd_kernel_traits, flash::copy, flash::gemm, flash::gemm_rs,
 * accumulator layouts, FP16 conversion, and FA's warp row reduction.
 * Framework source: MetaX-MACA/mcXformer aef88de756194e077460a1c8b8b341baa429259a.
 * This is an explicitly named finite-attribution operator, NOT model backward.
 * Q/K/V and U: FP16; MMA/softmax/row reductions: FP32; output: FP16.
 * Compact GQA inputs; query-head outputs retain the existing FP32 grouped reduction.
 * Endpoint means reuse existing RHS tile loads; no global Q/K midpoint buffers.
 * Two quadratic sweeps plus unmodified native FA dQ conversion. FP32 atomic
 * dQ summation follows native FA backward; default non-deterministic order.
 * No integral quadrature, no global N-by-N buffer, no model/package patch.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "flash_bwd_preprocess_kernel.h"

using namespace cute;
using FiniteTraits=Flash_fwd_kernel_traits<128,32,32,2,false,false,mctlass::half_t>;

// Native FA backward layouts for storing and transposing centered FP16 weights.
using TransposeTraits=Flash_bwd_kernel_traits<128,32,32,2,1,1,1>;
// Native convert_dQ uses the original 4-warp accumulation conversion layout.
using ConvertTraits=Flash_bwd_kernel_traits<128,32,32,4>;

// Explicit argument adapter to the unmodified templated convert_dQ routine.
// This is NOT the installed FA private ABI and needs no ATen generator header.
struct FiniteDqConvertParams {
    const int *cu_seqlens_q,*cu_seqlens_k,*seqused_k;
    const void *knew_ptr;
    bool is_seqlens_k_cumulative;
    int seqlen_q,seqlen_k,seqlen_knew,seqlen_q_rounded,h,h_k,d,d_rounded;
    void *dq_ptr,*dq_accum_ptr;
    int64_t dq_batch_stride,dq_row_stride,dq_head_stride,dq_accum_split_stride;
    float scale_softmax_rp_dropout;
};

struct FiniteParams {
    const mctlass::half_t *q0,*k0,*q1,*k1,*v0,*u;
    const float *lse0,*lse1;
    float *tau,*center;
    mctlass::half_t *dq,*dk,*dv;
    float *dq_accum;
    int batch,heads,kv_heads,length;
    float scale;
};

// Phase 0: row normalization and finite center.
// Phase 1: query-owned Q multiplier. Phase 2: key-owned K/V multipliers.
template<int Phase,typename Traits=FiniteTraits>
__global__ void deltatrace_fa_finite_p1_kernel_two_sweeps(FiniteParams p) {
    using E=typename Traits::Element;
    constexpr int M=Traits::kBlockM,N=Traits::kBlockN,D=Traits::kHeadDim;
    const int tid=threadIdx.x, bh=blockIdx.y, row0=blockIdx.x*M;
    static_assert(Phase==0 || Phase==2);
    const int length=p.length;
    const int rounded=((length+31)/32)*32;
    if constexpr(Phase==0) {
        // Same linear dQ accumulation workspace used by native FA backward.
        // Every row, including rounded padding, has exactly one initializer.
        for(int i=tid;i<M*D;i+=Traits::kNThreads) {
            const int row=row0+i/D,d=i%D;
            p.dq_accum[((int64_t(bh/p.heads)*rounded+row)*p.heads+bh%p.heads)*D+d]=0.f;
        }
    }
    const int64_t head_offset=int64_t(bh)*length*D;
    // Same GQA mapping as pinned FA flash_fwd_kernel.h: bidh / h_h_k_ratio.
    const int kv_bh=(bh/p.heads)*p.kv_heads+(bh%p.heads)/(p.heads/p.kv_heads);
    const int64_t kv_offset=int64_t(kv_bh)*length*D;
    const int64_t pair_left_offset=Phase==2?kv_offset:head_offset;
    const int64_t pair_right_offset=Phase==2?head_offset:kv_offset;
    extern __shared__ char scratch[];
    auto sA=make_tensor(make_smem_ptr(reinterpret_cast<E*>(scratch)),typename Traits::SmemLayoutQ{});
    auto sB=make_tensor(sA.data()+size(sA),typename Traits::SmemLayoutKV{});
    auto sBt=make_tensor(sB.data(),typename Traits::SmemLayoutVtransposed{});
    auto sBtPlain=make_tensor(sB.data(),typename Traits::SmemLayoutVtransposedNoSwizzle{});
    // Retain the FP16 midpoint in registers until the score products finish.
    // Their shared B tile is then dead and is reused for the multiplier GEMM.

    typename Traits::GmemTiledCopyQKV global_copy;
    auto global_thread=global_copy.get_thread_slice(tid);
    auto toA=global_thread.partition_D(sA);
    auto toB=global_thread.partition_D(sB);
    auto firstEndpointB=make_fragment_like(toB);
    auto firstEndpointA=make_fragment_like(toA);
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

    const int begin=Phase==2 ? row0 : 0;
    const int end=Phase==2 ? length : min(length,row0+M);
    for(int col0=begin;col0<end;col0+=N) {
        auto pair=[&](const E *left,const E *right,auto &acc,int endpoint) {
            __syncthreads();
            auto gA=make_tensor(make_gmem_ptr(left+pair_left_offset+int64_t(row0)*D),
                               Shape<Int<M>,Int<D>>{},Stride<Int<D>,_1>{});
            auto gB=make_tensor(make_gmem_ptr(right+pair_right_offset+int64_t(col0)*D),
                               Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
            auto fromGA=global_thread.partition_S(gA);
            auto fromGB=global_thread.partition_S(gB);
            flash::copy<false,true,true>(global_copy,fromGA,toA,coordsA,predA,length-row0);
            flash::copy<false,true,true>(global_copy,fromGB,toB,coordsB,predB,length-col0);
            __syncthreads();
            if constexpr(Phase!=0) {
                if(endpoint==0) {cute::copy(toB,firstEndpointB);cute::copy(toA,firstEndpointA);}
                if(endpoint==1) {
                    #pragma unroll
                    for(int i=0;i<size(firstEndpointB);++i)
                        firstEndpointB(i)=E((float(firstEndpointB(i))+float(toB(i)))*0.5f);
                    #pragma unroll
                    for(int i=0;i<size(firstEndpointA);++i)
                        firstEndpointA(i)=E((float(firstEndpointA(i))+float(toA(i)))*0.5f);
                }
            }
            clear(acc);
            flash::gemm(acc,regA,regB,fromA,fromB,mma,copyA,copyB,threadA,threadB);
        };
        auto a0=partition_fragment_C(mma,Shape<Int<M>,Int<N>>{});
        auto a1=make_fragment_like(a0);
        auto at=make_fragment_like(a0);
        if constexpr(Phase==2) {
            pair(p.k0,p.q0,a0,0);pair(p.k1,p.q1,a1,1);pair(p.v0,p.u,at,-1);
        } else {
            pair(p.q0,p.k0,a0,0);pair(p.q1,p.k1,a1,1);pair(p.u,p.v0,at,-1);
        }
        #pragma unroll
        for(int i=0;i<size(a0);++i) {
            const int r=row0+get<0>(coords(i)),c=col0+get<1>(coords(i));
            const int query=Phase==2?c:r,key=Phase==2?r:c;
            float lm=0.f,actual_p1=0.f;
            if(query<length && key<length && key<=query) {
                const int64_t pos=int64_t(bh)*length+query;
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
            auto multiply=[&](auto &weights,const E *values,auto &acc,bool shared_mean) {
                __syncthreads();
                if(shared_mean) {
                    cute::copy(firstEndpointB,toB);
                } else {
                auto gB=make_tensor(make_gmem_ptr(values+(Phase==1?kv_offset:head_offset)+int64_t(col0)*D),
                                   Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
                auto fromGB=global_thread.partition_S(gB);
                flash::copy<false,true,true>(global_copy,fromGB,toB,coordsB,predB,length-col0);
                }
                __syncthreads();
                CONVERT_TENSOR_TYPE(float,E,weights,half_weights)
                auto regWeights=make_tensor(half_weights.data(),weights.layout());
                flash::gemm_rs(acc,regWeights,regBt,fromBt,mma,copyBt,threadBt);
            };
            if constexpr(Phase==1) multiply(a0,nullptr,accQ,true);
            else {
                multiply(a0,nullptr,accQ,true);multiply(a1,p.u,accV,false);
                // Native flash_bwd_kernel.h pattern: store centered FP16 dS,
                // view its transpose in shared memory, then use flash::gemm.
                // Here a0 is finite L*(U V0^T-center), not a model gradient.
                __syncthreads();
                auto sW=make_tensor(sB.data()+size(sB),typename TransposeTraits::SmemLayoutPdS{});
                auto sWt=make_tensor(sW.data(),typename TransposeTraits::SmemLayoutPdStransposed{});
                auto sWtPlain=make_tensor(sW.data(),typename TransposeTraits::SmemLayoutPdStransposedNoSwizzle{});
                auto writeW=make_tiled_copy_C(typename TransposeTraits::SmemCopyAtomPdS{},mma);
                auto writeThread=writeW.get_thread_slice(tid);
                auto destW=writeThread.partition_D(sW);
                CONVERT_TENSOR_TYPE(float,E,a0,half_centered)
                auto weights=make_tensor(half_centered.data(),a0.layout());
                auto sourceW=writeThread.retile_S(weights);
                cute::copy(writeW,sourceW,destW);
                // The original A tile is dead. Reuse it for the K midpoint.
                cute::copy(firstEndpointA,toA);
                auto sAt=make_tensor(sA.data(),typename Traits::SmemLayoutVtransposed{});
                auto sAtPlain=make_tensor(sA.data(),typename Traits::SmemLayoutVtransposedNoSwizzle{});
                auto readW=make_tiled_copy_A(typename Traits::SmemCopyAtomTransposed{},mma);
                auto readK=make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{},mma);
                auto wt=readW.get_thread_slice(tid);auto kt=readK.get_thread_slice(tid);
                auto wsrc=wt.partition_S(sWt);auto ksrc=kt.partition_S(sAt);
                auto wreg=mma_thread.partition_fragment_A(sWtPlain);
                auto kreg=mma_thread.partition_fragment_B(sAtPlain);
                auto partialQ=partition_fragment_C(mma,Shape<Int<N>,Int<D>>{});
                clear(partialQ);__syncthreads();
                flash::gemm(partialQ,wreg,kreg,wsrc,ksrc,mma,readW,readK,wt,kt);
                // Native FA seq-k-parallel backward uses FP32 atomic dQ
                // accumulation; summation order can vary, as in default FA.
                auto qcoords=mma_thread.partition_C(make_identity_tensor(Shape<Int<N>,Int<D>>{}));
                #pragma unroll
                for(int i=0;i<size(partialQ);++i) {
                    const int query=col0+get<0>(qcoords(i)),dim=get<1>(qcoords(i));
                    if(query<length) {
                        const int64_t pos=((int64_t(bh/p.heads)*rounded+query)*p.heads+bh%p.heads)*D+dim;
                        atomicAdd(&p.dq_accum[pos],partialQ(i));
                    }
                }
            }
        }
    }
    if constexpr(Phase==0) {
        flash::quadreduce_sum(denominator);flash::quadreduce_sum(numerator);
        if(get<1>(rowcoords(0))==0) {
            #pragma unroll
            for(int i=0;i<size(denominator);++i) {
                const int r=row0+get<0>(rowcoords(i));
                if(r<length) {
                    p.tau[int64_t(bh)*length+r]=denominator(i);
                    p.center[int64_t(bh)*length+r]=numerator(i)/denominator(i);
                }
            }
        }
    } else {
        auto outcoords=mma_thread.partition_C(make_identity_tensor(Shape<Int<M>,Int<D>>{}));
        E *out=Phase==1?p.dq:p.dk;
        #pragma unroll
        for(int i=0;i<size(accQ);++i) {
            const int r=row0+get<0>(outcoords(i)),d=get<1>(outcoords(i));
            if(r<length) {
                const int64_t pos=head_offset+int64_t(r)*D+d;
                out[pos]=E(accQ(i)*p.scale);
                if constexpr(Phase==2) p.dv[pos]=E(accV(i));
            }
        }
    }
}

__global__ void deltatrace_fa_finite_convert_dq(FiniteDqConvertParams p) {
    // Unmodified vendor FA postprocessing, including FP32 scaling and FP16 store.
    flash::convert_dQ<ConvertTraits>(p,1);
}

extern "C" int deltatrace_fa_finite_p1_two_sweeps(
    const void *q0,const void *k0,const void *q1,const void *k1,const void *v0,
    const void *u,const void *lse0,const void *lse1,
    void *tau,void *center,void *dq,void *dk,void *dv,void *dq_accum,int batch,int heads,int kv_heads,int length,
    float scale,void *stream_ptr) {
    using E=mctlass::half_t;
    if(batch<1||heads<1||kv_heads<1||heads%kv_heads!=0||length<1)return -1;
    FiniteParams p{(const E*)q0,(const E*)k0,(const E*)q1,(const E*)k1,(const E*)v0,
        (const E*)u,(const float*)lse0,(const float*)lse1,
        (float*)tau,(float*)center,(E*)dq,(E*)dk,(E*)dv,(float*)dq_accum,batch,heads,kv_heads,length,scale};
    dim3 grid((length+FiniteTraits::kBlockM-1)/FiniteTraits::kBlockM,batch*heads);
    constexpr int shared=(size(typename FiniteTraits::SmemLayoutQ{})+size(typename FiniteTraits::SmemLayoutKV{}))*sizeof(E);
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    deltatrace_fa_finite_p1_kernel_two_sweeps<0><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    constexpr int weights_shared=size(typename TransposeTraits::SmemLayoutPdS{})*sizeof(E);
    deltatrace_fa_finite_p1_kernel_two_sweeps<2><<<grid,FiniteTraits::kNThreads,shared+weights_shared,stream>>>(p);
    error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    FiniteDqConvertParams convert{};
    convert.h=heads;convert.h_k=kv_heads;convert.seqlen_q=length;convert.seqlen_k=length;
    convert.seqlen_q_rounded=((length+31)/32)*32;convert.d=128;convert.d_rounded=128;
    convert.dq_ptr=dq;convert.dq_accum_ptr=dq_accum;
    convert.dq_batch_stride=int64_t(heads)*length*128;
    convert.dq_head_stride=int64_t(length)*128;convert.dq_row_stride=128;
    convert.scale_softmax_rp_dropout=scale;
    dim3 convert_grid((length+31)/32,heads,batch);
    constexpr int convert_shared=size(typename ConvertTraits::SmemLayoutdQ{})*sizeof(E);
    deltatrace_fa_finite_convert_dq<<<convert_grid,ConvertTraits::kNThreads,convert_shared,stream>>>(convert);
    return int(cudaGetLastError());
}
