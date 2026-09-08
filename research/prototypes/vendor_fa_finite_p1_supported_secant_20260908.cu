/* DeltaTrace finite P1 extension using the unmodified MetaX FA framework.
 * Framework portions Copyright (c) 2024, Tri Dao. BSD-3-Clause license retained.
 * Reuses Flash_fwd_kernel_traits, flash::copy, flash::gemm, flash::gemm_rs,
 * accumulator layouts, BF16 conversion, and FA's warp row reduction.
 * Framework source: MetaX-MACA/mcXformer aef88de756194e077460a1c8b8b341baa429259a.
 * This is an explicitly named finite-attribution operator, NOT model backward.
 * Q/K/V and U: BF16; MMA/softmax/row reductions: FP32; output: BF16.
 * Compact GQA inputs; query-head outputs retain the existing FP32 grouped reduction.
 * Endpoint means reuse existing RHS tile loads; no global Q/K midpoint buffers.
 * D256; right-padding lengths; full padding tiles skip arithmetic and write zeros.
 * No integral quadrature, no global N-by-N buffer, no model/package patch.
 * ISOLATED CANDIDATE: normalized input-endpoint softmax Jacobian plus the
 * probability-supported secant correction. Original QK midpoint
 * and raw P1 value branch retained. Real-arithmetic proof is in the protocol.
 * FP32 denominator is a sum of nonnegative raw-LSE Jeffreys terms using
 * the original expm1f logmean expression. Normalized direction/target and
 * raw denominator differ by explicitly audited default-precision row masses.
 * No claim of exact normalized Jeffreys or bit-exact secant conservation.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"

using namespace cute;
using FiniteTraits=Flash_fwd_kernel_traits<256,32,32,2,false,false,mctlass::bfloat16_t>;

enum SupportedRow {SAnchor,GAnchor,GMean,InvP0,InvP1,Kappa,DRaw,Numerator,P0Sum,P1Sum,
               RawTarget,NormalizedTarget,EndpointPositive,EndpointNegative,BaseContraction,
               DNorm,RowSum,OscG,DirectionRowSum,CorrectionPositive,CorrectionNegative,RowCount};
struct SupportedSecantParams {
    const void *q0,*k0,*q1,*k1,*v0,*u;
    const float *lse0,*lse1;
    float *rows;
    void *dq,*dk,*dv;
    const int *valid_lengths;
    int batch,heads,kv_heads,length;
    float scale;
};

// Phase 0: row normalization and finite center.
// Phase 1: query-owned Q multiplier. Phase 2: key-owned K/V multipliers.
template<int Phase,typename Traits=FiniteTraits>
__global__ void deltatrace_fa_finite_p1_supported_secant_kernel(SupportedSecantParams p) {
    using E=typename Traits::Element;
    // Like native Flash_fwd_params: opaque storage is typed inside each
    // compilation pass. The vendor traits use half_t on the host pass and
    // BF16 on the actual MACA device pass; do not override the vendor header.
    const E *q0=static_cast<const E*>(p.q0), *k0=static_cast<const E*>(p.k0);
    const E *q1=static_cast<const E*>(p.q1), *k1=static_cast<const E*>(p.k1);
    const E *v0=static_cast<const E*>(p.v0), *u=static_cast<const E*>(p.u);
    E *dq=static_cast<E*>(p.dq), *dk=static_cast<E*>(p.dk), *dv=static_cast<E*>(p.dv);
    constexpr int M=Traits::kBlockM,N=Traits::kBlockN,D=Traits::kHeadDim;
    const int tid=threadIdx.x, bh=blockIdx.y, row0=blockIdx.x*M;
    // Padded length remains the storage stride; valid bounds are per sample.
    const int length=p.length, valid=p.valid_lengths[bh/p.heads];
    const int64_t row_stride=int64_t(p.batch)*p.heads*length;
    const int64_t head_offset=int64_t(bh)*length*D;
    // Same GQA mapping as pinned FA flash_fwd_kernel.h: bidh / h_h_k_ratio.
    const int kv_bh=(bh/p.heads)*p.kv_heads+(bh%p.heads)/(p.heads/p.kv_heads);
    const int64_t kv_offset=int64_t(kv_bh)*length*D;
    const int64_t pair_left_offset=Phase==2?kv_offset:head_offset;
    const int64_t pair_right_offset=Phase==2?head_offset:kv_offset;
    // Every padding output is written, including entire empty tail tiles.
    // This branch is uniform within the block, before any synchronization.
    if(row0>=valid) {
        if constexpr(Phase==0) {
            for(int j=tid;j<M;j+=blockDim.x) if(row0+j<length) {
                const int64_t pos=int64_t(bh)*length+row0+j;
                for(int state=0;state<RowCount;++state) p.rows[state*row_stride+pos]=0.f;
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
    // Retain the BF16 midpoint in registers until the score products finish.
    // Their shared B tile is then dead and is reused for the multiplier GEMM.

    typename Traits::GmemTiledCopyQKV global_copy;
    auto global_thread=global_copy.get_thread_slice(tid);
    auto toA=global_thread.partition_D(sA);
    auto toB=global_thread.partition_D(sB);
    auto firstEndpointB=make_fragment_like(toB);
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
    auto sum_p0=make_fragment_like(denominator),sum_p1=make_fragment_like(denominator);
    auto sum_g0=make_fragment_like(denominator),sum_g1=make_fragment_like(denominator);
    auto sum_s1=make_fragment_like(denominator),sum_gs1=make_fragment_like(denominator);
    auto s_anchor=make_fragment_like(denominator),g_anchor=make_fragment_like(denominator);
    auto raw_divergence_sum=make_fragment_like(denominator);
    auto g_max=make_fragment_like(denominator),g_negmax=make_fragment_like(denominator);
    auto dnorm_sum=make_fragment_like(denominator),row_sum=make_fragment_like(denominator);
    auto direction_sum=make_fragment_like(denominator);
    auto correction_positive=make_fragment_like(denominator),correction_negative=make_fragment_like(denominator);
    clear(sum_p0);clear(sum_p1);clear(sum_g0);clear(sum_g1);clear(sum_s1);clear(sum_gs1);
    clear(s_anchor);clear(g_anchor);clear(raw_divergence_sum);
    clear(dnorm_sum);clear(row_sum);clear(direction_sum);clear(correction_positive);clear(correction_negative);
    #pragma unroll
    for(int i=0;i<size(g_max);++i) {g_max(i)=-INFINITY;g_negmax(i)=-INFINITY;}
    auto accQ=partition_fragment_C(mma,Shape<Int<M>,Int<D>>{});
    auto accV=make_fragment_like(accQ);
    clear(accQ);clear(accV);

    const int begin=Phase==2 ? row0 : 0;
    const int end=Phase==2 ? valid : min(valid,row0+M);
    for(int col0=begin;col0<end;col0+=N) {
        auto pair=[&](const E *left,const E *right,auto &acc,int endpoint) {
            __syncthreads();
            auto gA=make_tensor(make_gmem_ptr(left+pair_left_offset+int64_t(row0)*D),
                               Shape<Int<M>,Int<D>>{},Stride<Int<D>,_1>{});
            auto gB=make_tensor(make_gmem_ptr(right+pair_right_offset+int64_t(col0)*D),
                               Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
            auto fromGA=global_thread.partition_S(gA);
            auto fromGB=global_thread.partition_S(gB);
            flash::copy<false,true,true>(global_copy,fromGA,toA,coordsA,predA,valid-row0);
            flash::copy<false,true,true>(global_copy,fromGB,toB,coordsB,predB,valid-col0);
            __syncthreads();
            if constexpr(Phase!=0) {
                if(endpoint==0) cute::copy(toB,firstEndpointB);
                if(endpoint==1) {
                    #pragma unroll
                    for(int i=0;i<size(firstEndpointB);++i)
                        firstEndpointB(i)=E((float(firstEndpointB(i))+float(toB(i)))*0.5f);
                }
            }
            clear(acc);
            flash::gemm(acc,regA,regB,fromA,fromB,mma,copyA,copyB,threadA,threadB);
        };
        auto a0=partition_fragment_C(mma,Shape<Int<M>,Int<N>>{});
        auto a1=make_fragment_like(a0);
        auto at=make_fragment_like(a0);
        auto delta_score=make_fragment_like(a0);
        auto raw_divergence=make_fragment_like(a0),delta_prob=make_fragment_like(a0);
        if constexpr(Phase==2) {
            pair(k0,q0,a0,0);pair(k1,q1,a1,1);pair(v0,u,at,-1);
        } else {
            pair(q0,k0,a0,0);pair(q1,k1,a1,1);pair(u,v0,at,-1);
        }
        #pragma unroll
        for(int i=0;i<size(a0);++i) {
            const int r=row0+get<0>(coords(i)),c=col0+get<1>(coords(i));
            const int query=Phase==2?c:r,key=Phase==2?r:c;
            float actual_p0=0.f,actual_p1=0.f;
            delta_score(i)=0.f;raw_divergence(i)=0.f;delta_prob(i)=0.f;
            if(query<valid && key<valid && key<=query) {
                const int64_t pos=int64_t(bh)*length+query;
                const float lp0=a0(i)*p.scale-p.lse0[pos];
                const float lp1=a1(i)*p.scale-p.lse1[pos];
                delta_score(i)=(a1(i)-a0(i))*p.scale;
                actual_p0=__builtin_exp2f(lp0*float(M_LOG2E));
                actual_p1=__builtin_exp2f(lp1*float(M_LOG2E));
                if constexpr(Phase==0) {
                    const float distance=fabsf(lp1-lp0);
                    // exp(max lp) * (1-exp(-distance)) * distance >= 0;
                    // same raw probabilities and expm1f primitive as original LM.
                    raw_divergence(i)=(lp1>=lp0?actual_p1:actual_p0)*(-expm1f(-distance))*distance;
                } else {
                    const float centered_g=(at(i)-p.rows[GAnchor*row_stride+pos])-p.rows[GMean*row_stride+pos];
                    delta_prob(i)=actual_p1*p.rows[InvP1*row_stride+pos]-actual_p0*p.rows[InvP0*row_stride+pos];
                    a0(i)=query==0 ? 0.f : actual_p1*p.rows[InvP1*row_stride+pos]*centered_g
                          +p.rows[Kappa*row_stride+pos]*delta_prob(i);
                }
            } else if constexpr(Phase!=0) a0(i)=0.f;
            if constexpr(Phase==0) {a0(i)=actual_p0;a1(i)=actual_p1;}
            else a1(i)=actual_p1;
        }
        if constexpr(Phase==0) {
            flash::SumOp<float> sum;
            auto scratch_s=make_fragment_like(a0),scratch_g=make_fragment_like(a0);
            auto ss_rows=make_tensor(scratch_s.data(),flash::convert_layout_acc_rowcol(scratch_s.layout()));
            auto sg_rows=make_tensor(scratch_g.data(),flash::convert_layout_acc_rowcol(scratch_g.layout()));
            if(col0==0) {
                #pragma unroll
                for(int i=0;i<size(a0);++i) {
                    const int query=row0+get<0>(coords(i)),key=col0+get<1>(coords(i));
                    scratch_s(i)=query<valid && key==0 ? delta_score(i) : 0.f;
                    scratch_g(i)=query<valid && key==0 ? at(i) : 0.f;
                }
                flash::thread_reduce_<false>(ss_rows,s_anchor,sum);
                flash::thread_reduce_<false>(sg_rows,g_anchor,sum);
                flash::quadreduce_sum(s_anchor);flash::quadreduce_sum(g_anchor);
            }
            // Reuse the vendor accumulator row layout to broadcast row anchors.
            auto ds_rows=make_tensor(delta_score.data(),flash::convert_layout_acc_rowcol(delta_score.layout()));
            auto g_rows=make_tensor(at.data(),flash::convert_layout_acc_rowcol(at.layout()));
            auto p0_rows=make_tensor(a0.data(),flash::convert_layout_acc_rowcol(a0.layout()));
            auto p1_rows=make_tensor(a1.data(),flash::convert_layout_acc_rowcol(a1.layout()));
            auto coord_rows=make_tensor(coords.data(),flash::convert_layout_acc_rowcol(coords.layout()));
            auto divergence_rows=make_tensor(raw_divergence.data(),flash::convert_layout_acc_rowcol(raw_divergence.layout()));
            #pragma unroll
            for(int mi=0;mi<size<0>(ds_rows);++mi) {
                #pragma unroll
                for(int ni=0;ni<size<1>(ds_rows);++ni) {
                    const int query=row0+get<0>(coord_rows(mi,ni)),key=col0+get<1>(coord_rows(mi,ni));
                    const bool allowed=query<valid && key<valid && key<=query;
                    const float s=allowed?ds_rows(mi,ni)-s_anchor(mi):0.f;
                    const float g=allowed?g_rows(mi,ni)-g_anchor(mi):0.f;
                    raw_divergence_sum(mi)+=divergence_rows(mi,ni);
                    if(allowed) {g_max(mi)=fmaxf(g_max(mi),g_rows(mi,ni));g_negmax(mi)=fmaxf(g_negmax(mi),-g_rows(mi,ni));}
                    sum_p0(mi)+=p0_rows(mi,ni);sum_p1(mi)+=p1_rows(mi,ni);
                    sum_g0(mi)+=p0_rows(mi,ni)*g;sum_g1(mi)+=p1_rows(mi,ni)*g;
                    sum_s1(mi)+=p1_rows(mi,ni)*s;sum_gs1(mi)+=p1_rows(mi,ni)*g*s;
                }
            }
        } else {
            if constexpr(Phase==1) {
                auto weights_rows=make_tensor(a0.data(),flash::convert_layout_acc_rowcol(a0.layout()));
                auto ds_rows=make_tensor(delta_score.data(),flash::convert_layout_acc_rowcol(delta_score.layout()));
                auto dp_rows=make_tensor(delta_prob.data(),flash::convert_layout_acc_rowcol(delta_prob.layout()));
                auto coord_rows=make_tensor(coords.data(),flash::convert_layout_acc_rowcol(coords.layout()));
                #pragma unroll
                for(int mi=0;mi<size<0>(weights_rows);++mi) {
                    const int query=row0+get<0>(rowcoords(mi));
                    const int64_t pos=int64_t(bh)*length+query;
                    #pragma unroll
                    for(int ni=0;ni<size<1>(weights_rows);++ni) {
                        const int key=col0+get<1>(coord_rows(mi,ni));
                        if(query<valid && key<valid && key<=query) {
                            const float credit=weights_rows(mi,ni)*ds_rows(mi,ni);
                            const float h=p.rows[Kappa*row_stride+pos]*dp_rows(mi,ni);
                            denominator(mi)+=fmaxf(credit,0.f);numerator(mi)+=fminf(credit,0.f);
                            dnorm_sum(mi)+=dp_rows(mi,ni)*ds_rows(mi,ni);
                            row_sum(mi)+=weights_rows(mi,ni);direction_sum(mi)+=dp_rows(mi,ni);
                            correction_positive(mi)+=fmaxf(h,0.f);correction_negative(mi)+=fminf(h,0.f);
                        }
                    }
                }
            }
            auto multiply=[&](auto &weights,const E *values,auto &acc,bool shared_mean) {
                __syncthreads();
                if(shared_mean) {
                    cute::copy(firstEndpointB,toB);
                } else {
                auto gB=make_tensor(make_gmem_ptr(values+(Phase==1?kv_offset:head_offset)+int64_t(col0)*D),
                                   Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{});
                auto fromGB=global_thread.partition_S(gB);
                flash::copy<false,true,true>(global_copy,fromGB,toB,coordsB,predB,valid-col0);
                }
                __syncthreads();
                CONVERT_TENSOR_TYPE(float,E,weights,half_weights)
                auto regWeights=make_tensor(half_weights.data(),weights.layout());
                flash::gemm_rs(acc,regWeights,regBt,fromBt,mma,copyBt,threadBt);
            };
            if constexpr(Phase==1) multiply(a0,nullptr,accQ,true);
            else {multiply(a0,nullptr,accQ,true);multiply(a1,u,accV,false);}
        }
    }
    if constexpr(Phase==0) {
        flash::quadreduce_sum(sum_p0);flash::quadreduce_sum(sum_p1);
        flash::quadreduce_sum(sum_g0);flash::quadreduce_sum(sum_g1);
        flash::quadreduce_sum(sum_s1);flash::quadreduce_sum(sum_gs1);
        flash::quadreduce_sum(raw_divergence_sum);
        flash::MaxOp<float> max_op;
        flash::quad_allreduce_(g_max,g_max,max_op);flash::quad_allreduce_(g_negmax,g_negmax,max_op);
        if(get<1>(rowcoords(0))==0) {
            #pragma unroll
            for(int i=0;i<size(sum_p0);++i) {
                const int r=row0+get<0>(rowcoords(i));
                if(r<length) {
                    const int64_t pos=int64_t(bh)*length+r;
                    if(r>=valid) {for(int state=0;state<RowCount;++state)p.rows[state*row_stride+pos]=0.f;}
                    else {
                        const float inv0=1.f/sum_p0(i),inv1=1.f/sum_p1(i),gmean=sum_g1(i)*inv1;
                        const float target=gmean-sum_g0(i)/sum_p0(i);
                        const float base=(sum_gs1(i)-gmean*sum_s1(i))*inv1;
                        const float correction=target-base;
                        const float kappa=raw_divergence_sum(i)>0.f && r>0 ? correction/raw_divergence_sum(i):0.f;
                        p.rows[SAnchor*row_stride+pos]=s_anchor(i);
                        p.rows[GAnchor*row_stride+pos]=g_anchor(i);p.rows[GMean*row_stride+pos]=gmean;
                        p.rows[InvP0*row_stride+pos]=inv0;p.rows[InvP1*row_stride+pos]=inv1;
                        p.rows[Kappa*row_stride+pos]=kappa;p.rows[DRaw*row_stride+pos]=raw_divergence_sum(i);
                        p.rows[Numerator*row_stride+pos]=correction;
                        p.rows[OscG*row_stride+pos]=g_max(i)+g_negmax(i);
                        p.rows[DNorm*row_stride+pos]=0.f;p.rows[RowSum*row_stride+pos]=0.f;
                        p.rows[DirectionRowSum*row_stride+pos]=0.f;
                        p.rows[CorrectionPositive*row_stride+pos]=0.f;p.rows[CorrectionNegative*row_stride+pos]=0.f;
                        p.rows[P0Sum*row_stride+pos]=sum_p0(i);p.rows[P1Sum*row_stride+pos]=sum_p1(i);
                        p.rows[RawTarget*row_stride+pos]=(sum_g1(i)-sum_g0(i))+g_anchor(i)*(sum_p1(i)-sum_p0(i));
                        p.rows[NormalizedTarget*row_stride+pos]=target;
                        p.rows[EndpointPositive*row_stride+pos]=0.f;p.rows[EndpointNegative*row_stride+pos]=0.f;
                        p.rows[BaseContraction*row_stride+pos]=base;
                    }
                }
            }
        }
    } else {
        if constexpr(Phase==1) {
            flash::quadreduce_sum(denominator);flash::quadreduce_sum(numerator);
            flash::quadreduce_sum(dnorm_sum);flash::quadreduce_sum(row_sum);flash::quadreduce_sum(direction_sum);
            flash::quadreduce_sum(correction_positive);flash::quadreduce_sum(correction_negative);
            if(get<1>(rowcoords(0))==0) {
                #pragma unroll
                for(int i=0;i<size(denominator);++i) {
                    const int r=row0+get<0>(rowcoords(i));
                    if(r<length) {
                        const int64_t pos=int64_t(bh)*length+r;
                        p.rows[EndpointPositive*row_stride+pos]=r<valid?denominator(i):0.f;
                        p.rows[EndpointNegative*row_stride+pos]=r<valid?numerator(i):0.f;
                        p.rows[DNorm*row_stride+pos]=r<valid?dnorm_sum(i):0.f;
                        p.rows[RowSum*row_stride+pos]=r<valid?row_sum(i):0.f;
                        p.rows[DirectionRowSum*row_stride+pos]=r<valid?direction_sum(i):0.f;
                        p.rows[CorrectionPositive*row_stride+pos]=r<valid?correction_positive(i):0.f;
                        p.rows[CorrectionNegative*row_stride+pos]=r<valid?correction_negative(i):0.f;
                    }
                }
            }
        }
        auto outcoords=mma_thread.partition_C(make_identity_tensor(Shape<Int<M>,Int<D>>{}));
        E *out=Phase==1?dq:dk;
        #pragma unroll
        for(int i=0;i<size(accQ);++i) {
            const int r=row0+get<0>(outcoords(i)),d=get<1>(outcoords(i));
            if(r<length) {
                const int64_t pos=head_offset+int64_t(r)*D+d;
                out[pos]=E(r<valid?accQ(i)*p.scale:0.f);
                if constexpr(Phase==2) dv[pos]=E(r<valid?accV(i):0.f);
            }
        }
    }
}

extern "C" int deltatrace_fa_finite_p1_supported_secant(
    const void *q0,const void *k0,const void *q1,const void *k1,const void *v0,
    const void *u,const void *lse0,const void *lse1,
    void *rows,void *dq,void *dk,void *dv,const void *valid_lengths,int batch,int heads,int kv_heads,int length,
    float scale,void *stream_ptr) {
    using E=mctlass::bfloat16_t;
    if(batch<1||heads<1||kv_heads<1||heads%kv_heads!=0||length<1)return -1;
    SupportedSecantParams p{(const E*)q0,(const E*)k0,(const E*)q1,(const E*)k1,(const E*)v0,
        (const E*)u,(const float*)lse0,(const float*)lse1,
        (float*)rows,(E*)dq,(E*)dk,(E*)dv,(const int*)valid_lengths,batch,heads,kv_heads,length,scale};
    dim3 grid((length+FiniteTraits::kBlockM-1)/FiniteTraits::kBlockM,batch*heads);
    constexpr int shared=(size(typename FiniteTraits::SmemLayoutQ{})+size(typename FiniteTraits::SmemLayoutKV{}))*sizeof(E);
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    deltatrace_fa_finite_p1_supported_secant_kernel<0><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    deltatrace_fa_finite_p1_supported_secant_kernel<1><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    deltatrace_fa_finite_p1_supported_secant_kernel<2><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    return int(cudaGetLastError());
}
