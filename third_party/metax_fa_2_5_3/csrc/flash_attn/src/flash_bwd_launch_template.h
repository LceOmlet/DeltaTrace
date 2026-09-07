/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#ifdef EXPORT_LIB
    #include <cuda.h>
    #include "flash_attn.h"
    #include "mcblas.h"
    using namespace mcFlashAttn;
    #define C10_CUDA_CHECK(EXPR)
    #define C10_CUDA_KERNEL_LAUNCH_CHECK()
#else
    #include "flash.h"
    #include <ATen/cuda/CUDAContext.h>
#endif

#include "static_switch.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"

#include "flash_bwd_preprocess_kernel_hdim128_32_32.h"
#include "flash_bwd_kernel_hdim128_32_32.h"

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    const bool is_32x32 = (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 32);
    if constexpr (is_32x32) {
        flash::compute_dot_do_o_hdim128_32_32<Clear_dQaccum, Kernel_traits>(params);
    } else {
        flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
    }
}

template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(const Flash_bwd_params params) {
    flash::clear_dKVaccum<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_M, bool Is_even_K>
__global__ void flash_bwd_dq_dk_dv_loop_kernel(const Flash_bwd_params params) {
    flash::compute_dq_dk_dv<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,bool Is_deterministic=false>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel(const Flash_bwd_params params) {
    static_assert(!(Is_causal && Is_local));

    const bool is_32x32 = (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 32);
    if constexpr (is_32x32) {
        flash::compute_dq_dk_dv_seqk_parallel_hdim128_32_32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_deterministic>(params);
    } else {
        flash::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K,Is_deterministic>(params);
    }
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    const bool is_32x32 = (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 32);
    if constexpr (is_32x32) {
        flash::convert_dQ_hdim128_32_32<Kernel_traits>(params, nsplits);
    } else {
        flash::convert_dQ<Kernel_traits>(params, nsplits);
    }
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    flash::convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.h, params.b);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {

        #if 0
        auto dprops = at::cuda::getCurrentDeviceProperties();
        gridDimx = (dprops->multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
        #endif

        const int multiProcessorCount = 104;
        gridDimx = (multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.h, params.b);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                            params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;

    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        BOOL_SWITCH(params.deterministic, Is_deterministic, [&] {
                            auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout, Is_causal,
                                        Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local &&
                                        Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_deterministic>;

                            if (smem_size_dq_dk_dv >= 48 * 1024)  {
                                C10_CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                            }
                            kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 32 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_dropout>(params, stream);
#endif
}

template<typename T>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) {
            if constexpr(!Is_dropout) {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, false, T>, Is_dropout>(params, stream);
            } else {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, false, T>, Is_dropout>(params, stream);
            }
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, false, T>, Is_dropout>(params, stream);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, 4, 4, 4, true, true, true, T>, Is_dropout>(params, stream);
        #if 0
        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
        } else {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
        }
        #endif
    });
}

template<typename T>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {

        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout>(params, stream);
    });
}

template<typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }

#ifdef __USE_128_32x32
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr (!Is_dropout) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout>(params, stream);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, 4, 2, 2, true, false, true, T>, Is_dropout>(params, stream);
        }
    });
#else

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 64, 4, 2, 2, 2, true, false, true, T>, Is_dropout>(params, stream);
    });
#endif

}

template<typename T>
void run_mha_bwd_hdim160(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 116 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, false, T>, Is_dropout>(params, stream);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, true, T>, Is_dropout>(params, stream);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 136 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, false, T>, Is_dropout>(params, stream);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, true, T>, Is_dropout>(params, stream);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim224(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, false, T>, Is_dropout>(params, stream);
    });
}

template<typename T>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 176 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, false, T>, Is_dropout>(params, stream);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, true, T>, Is_dropout>(params, stream);
        }
    });
}
