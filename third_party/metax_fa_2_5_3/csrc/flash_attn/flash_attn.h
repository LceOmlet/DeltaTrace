/*
 * Copyright(c) 2021 MetaX Integrated Circuits Co.,Ltd. All rights reserved.
 */

#pragma once

#include <mcr/mc_runtime.h>

#include <vector>

#ifndef MCFLASHAPI
#if defined(__GNUC__) && __GNUC__ >= 4
#define MCFLASHAPI __attribute__((visibility("default")))
#else
#define MCFLASHAPI
#endif
#endif

namespace mcFlashAttn {

struct PhiloxMacaState {
    PhiloxMacaState() = default;

    PhiloxMacaState(uint64_t seed, uint64_t offset) {
        seed_.val = seed;
        offset_.val = offset;
    }

    PhiloxMacaState(int64_t *seed, int64_t *offset_extragraph, uint32_t offset_intragraph) {
        seed_.ptr = seed;
        offset_.ptr = offset_extragraph;
        offset_intragraph_ = offset_intragraph;
        captured_ = true;
    }

    union Payload {
        uint64_t val;
        int64_t *ptr;
    };

    Payload seed_;
    Payload offset_;
    uint32_t offset_intragraph_ = 0;
    bool captured_ = false;
};

__device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(PhiloxMacaState arg) {
    if (arg.captured_) {
        return std::make_tuple(static_cast<uint64_t>(*arg.seed_.ptr),
                               static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
    } else {
        return std::make_tuple(arg.seed_.val, arg.offset_.val);
    }
}

struct Qkv_params {
    using index_t = int64_t;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    int h, h_k;

    int h_h_k_ratio;
};

struct Flash_fwd_params : public Qkv_params {
    void *__restrict__ o_ptr;
    void *__restrict__ oaccum_ptr;

    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    void *__restrict__ p_ptr;

    void *__restrict__ softmax_lse_ptr;
    void *__restrict__ softmax_lseaccum_ptr;

    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;

    float scale_softmax;
    float scale_softmax_log2;

    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;

    int *__restrict__ seqused_k;

    int *__restrict__ blockmask;

    void *__restrict__ knew_ptr;
    void *__restrict__ vnew_ptr;

    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    void *__restrict__ rotary_cos_ptr;
    void *__restrict__ rotary_sin_ptr;

    int *__restrict__ cache_batch_idx;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    float p_dropout;
    uint8_t p_dropout_in_uint8_t;

    float rp_dropout;
    float scale_softmax_rp_dropout;

    int window_size_left, window_size_right;

    PhiloxMacaState philox_args;

    uint64_t *rng_state;

    bool is_bf16;
    bool is_causal;

    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;

    void *__restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool has_attn_mask;
    void *__restrict__ attn_mask_ptr;
    index_t attn_mask_batch_stride;
};

struct Flash_bwd_params : public Flash_fwd_params {
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;

    int packed_seqlen;
};

template <typename T, int Headdim>
MCFLASHAPI void run_mha_fwd_(Flash_fwd_params &params, mcStream_t stream);

template <typename T, int Headdim>
MCFLASHAPI void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, mcStream_t stream);

template <typename T, int Headdim>
MCFLASHAPI void run_mha_bwd_(Flash_bwd_params &params, mcStream_t stream);

}  // namespace mcFlashAttn