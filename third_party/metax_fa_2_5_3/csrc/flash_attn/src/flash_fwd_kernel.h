/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>

#include <mctlass/mctlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"
#include "attn_mask.h"

#if 1
#define BUITIN_ASM(x) asm volatile(#x)
#else
#define BUITIN_ASM(x)
#endif

namespace flash {
#ifdef EXPORT_LIB
using namespace mcFlashAttn;
#endif
using namespace cute;

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh,
                                                       const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

#ifdef EXPORT_LIB
    auto seed_offset = unpack(params.philox_args);
#else
    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
#endif
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0
                                      : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q -
                                                     params.window_size_left) /
                                                        kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
        Tensor gLSE =
            make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                        Shape<Int<kBlockM>>{}, Stride<_1>{});

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                  binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSE(row) = INFINITY;
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded +
        (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        } else {
            flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                 smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Has_attn_mask) {
            Element *bias_ptr =
                reinterpret_cast<Element *>(params.attn_mask_ptr) + bidb * params.attn_mask_batch_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, bidh, params.scale_softmax, bias_ptr, binfo.actual_seqlen_q);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o<true, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o<false, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                 smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Has_attn_mask) {
            Element *bias_ptr =
                reinterpret_cast<Element *>(params.attn_mask_ptr) + bidb * params.attn_mask_batch_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, bidh, params.scale_softmax, bias_ptr, binfo.actual_seqlen_q);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);

        softmax.template softmax_rescale_o<false, Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);

    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSE(row) = lse(mi);
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                              binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN,
          bool Is_even_K, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt(const Params &params, const int bidb, const int bidh,
                                                           const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    Tensor tQKVrQKV_t = make_tensor<uint32_t>(make_shape(Int<Kernel_traits::kHeadDim / 4>{}));
    auto tQrQ = tQKVrQKV_t.data();
    auto tVrV = tQKVrQKV_t.data();
    auto tKrK = tQKVrQKV_t.data() + Kernel_traits::kHeadDim / 8;

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
#ifdef EXPORT_LIB
    auto seed_offset = unpack(params.philox_args);
#else
    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
#endif
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0
                                      : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q -
                                                     params.window_size_left) /
                                                        kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
        Tensor gLSE =
            make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                        Shape<Int<kBlockM>>{}, Stride<_1>{});

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                  binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSE(row) = INFINITY;
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded +
        (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, tQpQ,
                                                         binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    }

    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tQrQ, tQsQ);
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();

        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV);
        } else {
            flash::copy_global_to_reg<Is_even_MN, Is_even_K, true>(tVgV, tVrV, tKVcKV, tKVpKV,
                                                                   binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o<true, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o<false, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

        flash::copy_global_to_reg<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);

        softmax.template softmax_rescale_o<false, Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);

    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSE(row) = lse(mi);
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                              binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN,
          bool Is_even_K, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt_hdim64(const Params &params, const int bidb, const int bidh,
                                                                  const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    Tensor tQKVrQKV_t = make_tensor<uint32_t>(make_shape(Int<Kernel_traits::kHeadDim / 4>{}));
    auto tQrQ = tQKVrQKV_t.data();
    auto tVrV = tQKVrQKV_t.data();
    auto tKrK = tQKVrQKV_t.data() + Kernel_traits::kHeadDim / 8;

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

#ifdef EXPORT_LIB
    auto seed_offset = unpack(params.philox_args);
#else
    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
#endif
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0
                                      : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q -
                                                     params.window_size_left) /
                                                        kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
        Tensor gLSE =
            make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                        Shape<Int<kBlockM>>{}, Stride<_1>{});

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                  binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSE(row) = INFINITY;
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
    }

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded +
        (n_block_max - 1) * kBlockN;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::cp_async_wait<0>();
    flash::copy_reg_to_share(tQrQ, tQsQ);

    int n_block = n_block_max - 1;
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
    uint32_t cpy_offset[5];
    const uint32_t laneId = __lane_id();
    const uint32_t row = ((laneId >> 4) << 2) + (laneId & 0x3);
    const uint32_t col = ((laneId >> 2) & 0x3) << 1;
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        cpy_offset[i] = ((((col + i * 8) >> 2) ^ (row & 0x7)) << 2) + (col & 0x3) + row * 32;
    }
    cpy_offset[4] = (((laneId & 0x7) << 1) + ((laneId >> 5) << 10));
    const uint32_t tOrVt_stride = 4;

    const int v_row = ((laneId & 0x7) - (laneId >> 3));
    const int tVgV_offset = v_row * (int(params.v_row_stride / 2) - 4);
    const int tVsV_offset = v_row * 32;

    if (Kernel_traits::Is_Q_in_regs) {
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        __syncthreads();
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    __syncthreads();
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsKstart);
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        BUITIN_ASM(; stsKend);

        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

            flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        } else {
            flash::copy_global_to_reg_V<Is_even_MN, Is_even_K, true>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset,
                                                                     binfo.actual_seqlen_k - n_block * kBlockN);
        }
        __syncthreads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsVstart);
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);
        masking_step == 0 ? flash::copy_reg_to_share_V<Is_even_MN, Is_even_K>(tVrV, tVsV, tVsV_offset)
                          : flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset);
        BUITIN_ASM(; stsVend);
        if (n_block > n_block_min) {
            BUITIN_ASM(; ldgKstart);
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

            BUITIN_ASM(; ldgKend);
        }
        __syncthreads();

        masking_step == 0 ? softmax.template softmax_rescale_o<true, Is_causal || Is_local, false, true>(acc_s, acc_o, params.scale_softmax_log2)
                          : softmax.template softmax_rescale_o<false, Is_causal || Is_local, false, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        masking_step == 0
            ? flash::gemm_rs_hdim64<Is_even_MN, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset,
                                                           tOrVt_stride)
            : flash::gemm_rs_hdim64<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOrVt_stride);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

#pragma unroll
    for (; n_block > n_block_min; --n_block) {
        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsKn_blockstart);
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        BUITIN_ASM(; stsKn_blockend);

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

        flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        __syncthreads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsVn_blockstart);
        flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset);

        tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);
        BUITIN_ASM(; stsVn_blockend);

        softmax.template softmax_rescale_o<false, Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rs_hdim64<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOrVt_stride);
    }

    if (n_block == n_block_min) {
        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsKn_blockstart);
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        BUITIN_ASM(; stsKn_blockend);

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

        flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        __syncthreads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsVn_blockstart);
        flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset);

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);
        BUITIN_ASM(; stsVn_blockend);
        __syncthreads();

        softmax.template softmax_rescale_o<false, Is_local, false, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rs_hdim64<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOrVt_stride);
    }

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);

    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSE(row) = lse(mi);
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                              binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN,
          bool Is_even_K, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt_hdim128(const Params &params, const int bidb, const int bidh,
                                                                   const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    Tensor tQKVrQKV_t = make_tensor<uint32_t>(make_shape(Int<Kernel_traits::kHeadDim / 4>{}));
    auto tQrQ = tQKVrQKV_t.data();
    auto tVrV = tQKVrQKV_t.data();
    auto tKrK = tQKVrQKV_t.data() + Kernel_traits::kHeadDim / 8;

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

#ifdef EXPORT_LIB
    auto seed_offset = unpack(params.philox_args);
#else
    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
#endif
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0
                                      : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q -
                                                     params.window_size_left) /
                                                        kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
        Tensor gLSE =
            make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                        Shape<Int<kBlockM>>{}, Stride<_1>{});

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                  binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSE(row) = INFINITY;
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
    }

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded +
        (n_block_max - 1) * kBlockN;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::cp_async_wait<0>();
    flash::copy_reg_to_share(tQrQ, tQsQ);

    int n_block = n_block_max - 1;
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
    uint32_t cpy_offset[5];
    const uint32_t laneId = __lane_id();

    const uint32_t row = ((laneId >> 4) << 2) + (laneId & 0x3);
    const uint32_t col = ((laneId >> 2) & 0x3) << 1;

    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        cpy_offset[i] = ((((col + i * 8) >> 2) ^ (row & 0x7)) << 2) + (col & 0x3) + row * 32;
    }

    cpy_offset[4] = (((laneId & 0x7) << 1) + ((laneId >> 5) << 10));
    const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().layout_fn().stride())) / 2;
    const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride())) / 2;

    const int v_row = ((laneId & 0x7) - (laneId >> 3));
    const int tVgV_offset = v_row * (int(params.v_row_stride / 2) - 4);
    const int tVsV_offset = v_row * 32;

    if (Kernel_traits::Is_Q_in_regs) {
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        const int lane_id = __lane_id();
        const int col = (((lane_id & 0x7) ^ (lane_id >> 5)) << 3) + (((lane_id >> 4) & 0x1) << 2);
        __syncthreads();
        if constexpr (kBlockM == 128) {
            auto sm_ptr = reinterpret_cast<uint32_t *>(tSsQ.data().ptr_ - col);
            auto r_ptr = reinterpret_cast<uint32_t *>(tSrQ_copy_view.data());
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = (((lane_id & 0x7) ^ ((lane_id >> 5) + 2 * i)) << 2) + (((lane_id >> 4) & 0x1) << 1);
                auto ptr0 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i);
                auto s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col);
                ptr0[0] = s_ptr0[0];

                ptr0 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i + 2);
                s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col + 64 * 32);
                ptr0[0] = s_ptr0[0];

                auto ptr1 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i + 16);
                auto s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 128 * 32);
                ptr1[0] = s_ptr1[0];

                ptr1 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i + 16 + 2);
                s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 192 * 32);
                ptr1[0] = s_ptr1[0];
            }
        } else if constexpr (kBlockM == 64) {
            auto sm_ptr = reinterpret_cast<uint32_t *>(tSsQ.data().ptr_ - col);
            auto r_ptr = reinterpret_cast<uint32_t *>(tSrQ_copy_view.data());
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = (((lane_id & 0x7) ^ ((lane_id >> 5) + 2 * i)) << 2) + (((lane_id >> 4) & 0x1) << 1);
                auto ptr0 = reinterpret_cast<uint64_t *>(r_ptr + 2 * i);
                auto s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col);
                auto ptr1 = reinterpret_cast<uint64_t *>(r_ptr + 2 * i + 8);
                auto s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 64 * 32);
                ptr0[0] = s_ptr0[0];
                ptr1[0] = s_ptr1[0];
            }

        } else {
            cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        }
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    __syncthreads();
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsKstart);
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        BUITIN_ASM(; stsKend);

        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

            flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        } else {
            flash::copy_global_to_reg_V<Is_even_MN, Is_even_K, true>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset,
                                                                     binfo.actual_seqlen_k - n_block * kBlockN);
        }
        __syncthreads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsVstart);
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);
        masking_step > 0 ? flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset)
                         : flash::copy_reg_to_share_V<Is_even_MN, Is_even_K>(tVrV, tVsV, tVsV_offset);
        BUITIN_ASM(; stsVend);
        if (n_block > n_block_min) {
            BUITIN_ASM(; ldgKstart);
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);
            BUITIN_ASM(; ldgKend);
        }

        masking_step > 0
            ? softmax.template softmax_rescale_o<false, Is_causal || Is_local, true, kBlockM != 128 ? true : false>(
                  acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o<true, Is_causal || Is_local, true, kBlockM != 128 ? true : false>(
                  acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        masking_step > 0 ? flash::gemm_rs<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset,
                                                           tOsVt_stride, tOrVt_stride)
                         : flash::gemm_rs<Is_even_MN, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset,
                                                                 tOsVt_stride, tOrVt_stride);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

#pragma unroll
    for (; n_block > n_block_min; --n_block) {
        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsKn_blockstart);
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        BUITIN_ASM(; stsKn_blockend);

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

        flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        __syncthreads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsVn_blockstart);
        flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset);

        tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);
        BUITIN_ASM(; stsVn_blockend);
        constexpr bool Sync_in_softmax = (kBlockM != 128 || kBlockN != 64);
        if constexpr (!Sync_in_softmax) __syncthreads();

        softmax.template softmax_rescale_o<false, Is_local, Sync_in_softmax, kBlockM != 128 ? true : false>(acc_s, acc_o,
                                                                                           params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    if (n_block == n_block_min) {
        flash::cp_async_wait<0>();
        BUITIN_ASM(; stsKn_blockstart);
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        BUITIN_ASM(; stsKn_blockend);

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

        flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        __syncthreads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset);

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);

        if constexpr (kBlockM != 128 || kBlockN != 64) __syncthreads();
        softmax.template softmax_rescale_o<false, Is_local, false, kBlockM != 128 ? true : false>(acc_s, acc_o,
                                                                                           params.scale_softmax_log2);
        if constexpr (kBlockM == 128 && kBlockN == 64) __syncthreads();

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);

    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSE(row) = lse(mi);
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                              binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Split, bool Append_KV, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh,
                                                               const int m_block, const int n_split_idx,
                                                               const int num_n_splits) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyO =
        std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyOaccum, typename Kernel_traits::GmemTiledCopyO>;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo<!Is_even_MN> binfo(params, bidb);

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min =
        !Is_local
            ? n_split_idx * n_blocks_per_split
            : std::max(n_split_idx * n_blocks_per_split,
                       (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) /
                           kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }
    if (n_block_min >= n_block_max) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum =
            (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) *
            params.d_rounded;
        const index_t row_offset_lseaccum =
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) +
                          (Split ? row_offset_oaccum : row_offset_o)),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(
                                                         Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) +
                                                     row_offset_lseaccum),
                                       Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));

        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO,
                                                  binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSEaccum(row) = Split ? -INFINITY : INFINITY;
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;

    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table =
        params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset =
        block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k =
        block_table == nullptr
            ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache) +
                  (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
            : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride +
                  (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v =
        block_table == nullptr
            ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache) +
                  (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
            : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride +
                  (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));

    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    if constexpr (Append_KV) {
        const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN) * (params.rotary_dim / 2);
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor gCosCont =
            make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSinCont =
            make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);

        const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb) +
                                        ((n_block_max - 1) * kBlockN) * params.knew_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb) +
                                        ((n_block_max - 1) * kBlockN) * params.vnew_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.vnew_head_stride;

        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr) + row_offset_knew -
                                                 binfo.seqlen_k_cache * params.knew_row_stride),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.knew_row_stride, _1{}));

        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr) + row_offset_vnew -
                                                 binfo.seqlen_k_cache * params.vnew_row_stride),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.vnew_row_stride, _1{}));
        Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);
        Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);

        const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
        auto tKgK_data = tKgK.data();
        auto tVgV_data = tVgV.data();
        for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
            flash::copy_w_min_idx<Is_even_K>(tVgVnew, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN,
                                             binfo.seqlen_k_cache - n_block * kBlockN);
            tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
            if (params.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K>(tKgKnew, tKgK, tKVcKV, tKVpKV,
                                                 binfo.actual_seqlen_k - n_block * kBlockN,
                                                 binfo.seqlen_k_cache - n_block * kBlockN);
            } else {
                if (params.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K, false>(
                        tKgKnew, tKgK, tRgCos, tRgSin, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim);
                    tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K, false>(
                        tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim);
                    tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                }
            }
            tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                if (n_block > n_block_copy_min) {
                    const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                    const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                    const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                    const int block_table_offset_next =
                        (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                    const int table_diff = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
                    const int offset_diff = block_table_offset_next - block_table_offset_cur;
                    tVgV.data() = tVgV.data() + table_diff * params.v_batch_stride + offset_diff * params.v_row_stride;
                    tKgK.data() = tKgK.data() + table_diff * params.k_batch_stride + offset_diff * params.k_row_stride;
                }
            }
        }

        __syncthreads();
        tKgK.data() = tKgK_data;
        tVgV.data() = tVgV_data;
    }

    if (!Append_KV || params.rotary_dim == 0) {
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        const index_t row_offset_cossin =
            (binfo.seqlen_k_cache + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);

        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        if (params.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K>(tQgQ, tQsQ, tRgCos, tRgSin, tQcQ,
                                                      binfo.actual_seqlen_q - m_block * kBlockM, 0, params.d,
                                                      params.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K>(tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ,
                                                     binfo.actual_seqlen_q - m_block * kBlockM, 0, params.d,
                                                     params.rotary_dim);
        }
    }

    int n_block = n_block_max - 1;

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        if (masking_step > 0) {
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
                const int block_table_offset_cur =
                    (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
                tVgV.data() =
                    tVgV.data() +
                    (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride +
                    (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
            }
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        } else {
            flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q,
                    smem_thr_copy_K);

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);

        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block > n_block_min) {
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =
                    (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() =
                    tKgK.data() +
                    (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride +
                    (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        masking_step == 0 ? softmax.template softmax_rescale_o<true, Is_causal || Is_local || !Is_even_MN>(
                                acc_s, acc_o, params.scale_softmax_log2)
                          : softmax.template softmax_rescale_o<false, Is_causal || Is_local || !Is_even_MN>(
                                acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
            const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
            const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
            const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
            tVgV.data() =
                tVgV.data() +
                (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride +
                (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
        }
        flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q,
                    smem_thr_copy_K);

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =
                    (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() =
                    tKgK.data() +
                    (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride +
                    (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);
        softmax.template softmax_rescale_o<false, Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor lse = softmax.template normalize_softmax_lse<false, Split>(acc_o, params.scale_softmax);

    Tensor sOaccum =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{});

    using SmemTiledCopyO =
        std::conditional_t<!Split, typename Kernel_traits::SmemCopyAtomO, typename Kernel_traits::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);

    CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o, rO)
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);

    if constexpr (Split) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum =
        (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum =
        ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum =
        make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) +
                                  (Split ? row_offset_oaccum : row_offset_o)),
                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) +
                      row_offset_lseaccum),
        Shape<Int<kBlockM>>{}, Stride<_1>{});

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);

    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSEaccum(row) = lse(mi);
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));

    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO,
                                              binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params, int Rowblock_Parallel_Num = 1>
__forceinline__ __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;

    const int bidb = blockIdx.z;

    const int bidh = blockIdx.y;

    if constexpr (Has_attn_mask || Kernel_traits::kHeadDim >= 224) {
        flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                      Is_even_MN, Is_even_K, Return_softmax, Params>(params, bidb, bidh, m_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            const int m_block_max = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                          Is_even_MN, Is_even_K, Return_softmax, Params>(params, bidb, bidh,
                                                                                         m_block_max - m_block - 1);
        }
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_opt_hdim128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN,
                                                  Is_even_K, Return_softmax, Params>(params, bidb, bidh, m_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            const int m_block_max = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            flash::compute_attn_1rowblock_opt_hdim128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                      Is_even_MN, Is_even_K, Return_softmax, Params>(
                params, bidb, bidh, m_block_max - m_block - 1);
        }
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 64) {
        flash::compute_attn_1rowblock_opt_hdim64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN,
                                                 Is_even_K, Return_softmax, Params>(params, bidb, bidh, m_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            const int m_block_max = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            flash::compute_attn_1rowblock_opt_hdim64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                     Is_even_MN, Is_even_K, Return_softmax, Params>(
                params, bidb, bidh, m_block_max - m_block - 1);
        }
        return;
    }

    flash::compute_attn_1rowblock_opt<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K,
                                      Return_softmax, Params>(params, bidb, bidh, m_block);
    if constexpr (Rowblock_Parallel_Num == 2) {
        const int m_block_max = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
        flash::compute_attn_1rowblock_opt<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN,
                                          Is_even_K, Return_softmax, Params>(params, bidb, bidh,
                                                                             m_block_max - m_block - 1);
    }
}

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Split, bool Append_KV, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_hdim128(const Params &params, const int bidb,
                                                                       const int bidh, const int m_block,
                                                                       const int n_split_idx, const int num_n_splits) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    Tensor tQKVrQKV_t = make_tensor<uint32_t>(make_shape(Int<Kernel_traits::kHeadDim / 4>{}));
    auto tVrV = tQKVrQKV_t.data();
    auto tKrK = tQKVrQKV_t.data() + Kernel_traits::kHeadDim / 8;
    auto tGrKV = tQKVrQKV_t.data();
    auto tGrV = tVrV;
    auto tGrK = tKrK;

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyO =
        std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyOaccum, typename Kernel_traits::GmemTiledCopyO>;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo<!Is_even_MN> binfo(params, bidb);

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min =
        !Is_local
            ? n_split_idx * n_blocks_per_split
            : std::max(n_split_idx * n_blocks_per_split,
                       (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) /
                           kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }
    if (n_block_min >= n_block_max) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum =
            (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) *
            params.d_rounded;
        const index_t row_offset_lseaccum =
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) +
                          (Split ? row_offset_oaccum : row_offset_o)),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(
                                                         Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) +
                                                     row_offset_lseaccum),
                                       Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));

        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO,
                                                  binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSEaccum(row) = Split ? -INFINITY : INFINITY;
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;

    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table =
        params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset =
        block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k =
        block_table == nullptr
            ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache) +
                  (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
            : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride +
                  (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v =
        block_table == nullptr
            ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache) +
                  (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
            : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride +
                  (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));

    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    if constexpr (Append_KV) {
        const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN) * (params.rotary_dim / 2);
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor gCosCont =
            make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSinCont =
            make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.rotary_dim / 2, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);

        const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb) +
                                        ((n_block_max - 1) * kBlockN) * params.knew_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb) +
                                        ((n_block_max - 1) * kBlockN) * params.vnew_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.vnew_head_stride;

        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr) + row_offset_knew -
                                                 binfo.seqlen_k_cache * params.knew_row_stride),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.knew_row_stride, _1{}));

        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr) + row_offset_vnew -
                                                 binfo.seqlen_k_cache * params.vnew_row_stride),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.vnew_row_stride, _1{}));
        Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);
        Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);

        const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
        auto tKgK_data = tKgK.data();
        auto tVgV_data = tVgV.data();
        for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
            if (params.rotary_dim == 0) {
                flash::copy_w_min_idx_kv<Is_even_K>(tVgVnew, tKgKnew, tVgV, tKgK, tGrV, tGrK, tKVcKV, tKVpKV,
                                                    binfo.actual_seqlen_k - n_block * kBlockN,
                                                    binfo.seqlen_k_cache - n_block * kBlockN);
            } else {
                flash::copy_w_min_idx<Is_even_K>(tVgVnew, tVgV, tGrKV, tKVcKV, tKVpKV,
                                                 binfo.actual_seqlen_k - n_block * kBlockN,
                                                 binfo.seqlen_k_cache - n_block * kBlockN);
                if (params.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K, false>(
                        tKgKnew, tKgK, tRgCos, tRgSin, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim);
                    tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K, false>(
                        tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim);
                    tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                }
            }
            tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
            tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                if (n_block > n_block_copy_min) {
                    const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                    const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                    const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                    const int block_table_offset_next =
                        (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                    const int table_diff = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
                    const int offset_diff = block_table_offset_next - block_table_offset_cur;
                    tVgV.data() = tVgV.data() + table_diff * params.v_batch_stride + offset_diff * params.v_row_stride;
                    tKgK.data() = tKgK.data() + table_diff * params.k_batch_stride + offset_diff * params.k_row_stride;
                }
            }
        }

        __syncthreads();
        tKgK.data() = tKgK_data;
        tVgV.data() = tVgV_data;
    }

    if (!Append_KV || params.rotary_dim == 0) {
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        const index_t row_offset_cossin =
            (binfo.seqlen_k_cache + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);

        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        if (params.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K>(tQgQ, tQsQ, tRgCos, tRgSin, tQcQ,
                                                      binfo.actual_seqlen_q - m_block * kBlockM, 0, params.d,
                                                      params.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K>(tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ,
                                                     binfo.actual_seqlen_q - m_block * kBlockM, 0, params.d,
                                                     params.rotary_dim);
        }
    }

    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    if (Kernel_traits::Is_Q_in_regs) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    int n_block = n_block_max - 1;

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    const uint32_t laneId = __lane_id();
    const uint32_t row = ((laneId >> 4) << 2) + (laneId & 0x3);
    const uint32_t col = ((laneId >> 2) & 0x3) << 1;
    uint32_t cpy_offset[5];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        cpy_offset[i] = ((((col + i * 8) >> 2) ^ (row & 0x7)) << 2) + (col & 0x3) + row * 32;
    }
    cpy_offset[4] = (((laneId & 0x7) << 1) + ((laneId >> 5) << 10));
    const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().layout_fn().stride())) / 2;
    const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride())) / 2;

    const int v_row = ((laneId & 0x7) - (laneId >> 3));
    const int tVgV_offset = v_row * (int(params.v_row_stride / 2)) - v_row * 4;
    const int tVsV_offset = v_row * 32;

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();

        if (masking_step > 0) {
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
                const int block_table_offset_cur =
                    (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
                tVgV.data() =
                    tVgV.data() +
                    (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride +
                    (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
            }

            flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        } else {
            flash::copy_global_to_reg_V<Is_even_MN, Is_even_K, true>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset,
                                                                     binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);

        flash::cp_async_wait<0>();
        masking_step == 0 ? flash::copy_reg_to_share_V<Is_even_MN, Is_even_K>(tVrV, tVsV, tVsV_offset)
                          : flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset);
        __syncthreads();

        if (n_block > n_block_min) {
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =
                    (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() =
                    tKgK.data() +
                    (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride +
                    (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        masking_step == 0 ? softmax.template softmax_rescale_o<true, Is_causal || Is_local || !Is_even_MN>(
                                acc_s, acc_o, params.scale_softmax_log2)
                          : softmax.template softmax_rescale_o<false, Is_causal || Is_local || !Is_even_MN>(
                                acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        masking_step == 0 ? flash::gemm_rs<Is_even_MN, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset,
                                                                  tOsVt_stride, tOrVt_stride)
                          : flash::gemm_rs<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset,
                                                            tOsVt_stride, tOrVt_stride);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();

        if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
            const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
            const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
            const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
            tVgV.data() =
                tVgV.data() +
                (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride +
                (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
        }

        flash::copy_global_to_reg_V<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV, tVgV_offset);
        cute::cp_async_fence();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share_V<true, Is_even_K>(tVrV, tVsV, tVsV_offset);
        __syncthreads();
        if (n_block > n_block_min) {
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =
                    (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() =
                    tKgK.data() +
                    (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride +
                    (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);
        softmax.template softmax_rescale_o<false, Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rs<true, Is_even_K>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    Tensor lse = softmax.template normalize_softmax_lse<false, Split>(acc_o, params.scale_softmax);

    Tensor sOaccum =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{});

    using SmemTiledCopyO =
        std::conditional_t<!Split, typename Kernel_traits::SmemCopyAtomO, typename Kernel_traits::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);

    CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o, rO)
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);

    if constexpr (Split || Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum =
        (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum =
        ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum =
        make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) +
                                  (Split ? row_offset_oaccum : row_offset_o)),
                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) +
                      row_offset_lseaccum),
        Shape<Int<kBlockM>>{}, Stride<_1>{});

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);

    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSEaccum(row) = lse(mi);
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));

    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO,
                                              binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Split, bool Append_KV, typename Params>
__forceinline__ __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;

    const int bidb = Split ? blockIdx.y / params.h : blockIdx.z;

    const int bidh = Split ? blockIdx.y - bidb * params.h : blockIdx.y;
    const int n_split_idx = Split ? blockIdx.z : 0;
    const int num_n_splits = Split ? gridDim.z : 1;
    if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_splitkv_hdim128<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN,
                                                      Is_even_K, Split, Append_KV>(params, bidb, bidh, m_block,
                                                                                   n_split_idx, num_n_splits);
        return;
    }

    flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Split,
                                          Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}

template <typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
__forceinline__ __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = 256;
    ;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128 || kNThreads == 256, "We assume that each block has 128 or 256 threads");

    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum =
        make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                    Shape<Int<kMaxSplits>, Int<kBlockM>>{}, make_stride(params.b * params.h * params.seqlen_q, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
#pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < params.b * params.h * params.seqlen_q - bidx * kBlockM)
                               ? gLSEaccum(row, col)
                               : -INFINITY;
        if (row < kMaxSplits) {
            sLSE[row][col] = lse;
        }
    }

    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);

    static_assert(kRowsPerLoadTranspose <= 64);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
#pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
    }

    ElementAccum lse_max = lse_accum(0);
#pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) {
        lse_max = max(lse_max, lse_accum(l));
    }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;
    float lse_sum = expf(lse_accum(0) - lse_max);
#pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) {
        lse_sum += expf(lse_accum(l) - lse_max);
    }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);

    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;

    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
    }

#pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < params.num_splits && col < kBlockM) {
            sLSE[row][col] = expf(lse_accum(l) - lse_logsum);
        }
    }
    __syncthreads();

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{}, GmemLayoutAtomOaccum{},
                                                         Layout<Shape<_1, _4>>{}));
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});

    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) {
            tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d;
        }
    }

    for (int split = 0; split < params.num_splits; ++split) {
        flash::copy<false, Is_even_K>(gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum,
                                      params.b * params.h * params.seqlen_q - bidx * kBlockM);
#pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
#pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
#pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }
            }
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
    }

    CONVERT_TENSOR_TYPE(ElementAccum, Element, tOrO, rO)

#pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < params.b * params.h * params.seqlen_q) {
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;

            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride +
                         head_idx * params.o_head_stride + row * params.o_row_stride;
#pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                if (Is_even_K || tOpOaccum(k)) {
                    const int col = get<1>(tOcOaccum(0, m, k));
                    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col), Shape<Int<decltype(size<0>(rO))::value>>{},
                                            Stride<_1>{});

                    copy(rO(_, m, k), gO);
                }
            }
        }
    }
}

}  // namespace flash
