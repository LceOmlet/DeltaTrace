/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <mctlass/array.h>
#include <mctlass/mctlass.h>
#include <mctlass/numeric_conversion.h>
#include <mctlass/numeric_types.h>

namespace flash {

template <typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template <>
__forceinline__ __device__ uint32_t relu2<mctlass::half_t>(const uint32_t x) {
    uint32_t res;
#if defined(__MACA_ARCH__)
    //asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    auto y = *reinterpret_cast<__half2 const *>(&x);
    __half zero = __half(0);
    y.x = y.x > zero ? y.x : zero;
    y.y = y.y > zero ? y.y : zero;
    res = *reinterpret_cast<uint32_t *>(&y);
#else
    const uint32_t zero = 0u;
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__MACA_ARCH__)
template <>
__forceinline__ __device__ uint32_t relu2<mctlass::bfloat16_t>(const uint32_t x) {
    auto y = *reinterpret_cast<__maca_bfloat162 const *>(&x);
    __maca_bfloat16 zero = __maca_bfloat16(0);
    y.x = y.x > zero ? y.x : zero;
    y.y = y.y > zero ? y.y : zero;
    uint32_t res = *reinterpret_cast<uint32_t *>(&y);
    return res;
}
#endif

#if 0

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<mctlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<mctlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
    __device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template <>
struct Allreduce<64> {
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
#if 0
        constexpr int OFFSET = 32;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET / 2));
        return x;
#endif

        auto x1 = __shfl_xor_sync(uint64_t(-1), x, 48);
        auto x2 = __shfl_xor_sync(uint64_t(-1), x, 32);
        auto x3 = __shfl_xor_sync(uint64_t(-1), x, 16);
        return op(op(op(x, x1), x2), x3);
    }
};

template <>
struct Allreduce<2> {
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, 1));
        return x;
    }
};

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                     Tensor4 const &tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
                                     TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    }
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
            if (!B_in_regs) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_opt(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                         Tensor4 const &tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
                                         TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A,
                                         ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    }
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
            if (!B_in_regs) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
        }

        __builtin_mxc_schedbound_begin();
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        __builtin_mxc_schedbound_end();
    }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                        TiledMma tiled_mma, TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);

    return make_layout(make_layout(cute::Layout<_1>{}, get<1>(acc_layout)),
                       make_layout(get<0>(acc_layout), get<2>(acc_layout)));
};

template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    mctlass::NumericArrayConverter<To_type, From_type, numel> convert_op;

    auto frag = convert_op(*reinterpret_cast<const mctlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

#define CONVERT_TENSOR_TYPE(type_s, type_d, tensor_s, tensor_d)                                                                         \
    constexpr int tensor_d##_numel = decltype(size(tensor_s))::value;                                                                   \
    mctlass::NumericArrayConverter<type_d, type_s, tensor_d##_numel > tensor_d##_convert_op;                                            \
    auto tensor_d##_frag = tensor_d##_convert_op(*reinterpret_cast<const mctlass::Array<type_s, tensor_d##_numel> *>(tensor_s.data())); \
    Tensor tensor_d = make_tensor(make_rmem_ptr<type_d>(&tensor_d##_frag), tensor_s.layout());

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;

    Tensor tensor_uint32 = recast<uint32_t>(tensor);
#pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, mctlass::half_t> || std::is_same_v<To_type, mctlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if 0

    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
#pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else

    CONVERT_TENSOR_TYPE(From_type, To_type, tensor, out)
    flash::relu_(out);
#endif
    return out;
}

template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <>
CUTE_HOST_DEVICE void cp_async_wait<0>() {
#if defined(__MACA_ARCH__)
    __builtin_mxc_arrive(64);
#endif
}

__forceinline__ __device__ void cp_sync() {
    __builtin_mxc_arrive(4096 + 64);
    __builtin_mxc_barrier();
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                     Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                     Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

template <bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &identity_MN,
                                               Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0,
                                               const int min_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

template <bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> &D,
                                               uint32_t *reg, Tensor<Engine2, Layout2> const &identity_MN,
                                               Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0,
                                               const int min_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                const int idx = (m * size<2>(S) + k) * 4;
                if (Is_even_K || predicate_K(k)) {
                    cute::copy_global_to_reg(S(_, m, k), reg + idx);
                }
            }
        }
    }
    flash::cp_async_wait<0>();

#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                auto reg_ptr = reg + (m * size<2>(S) + k) * 4;
                if (Is_even_K || predicate_K(k)) {
                    auto dst_ptr = reinterpret_cast<uint32_t *>(D(_, m, k).data().ptr_);
                    dst_ptr[0] = reg_ptr[0];
                    dst_ptr[1] = reg_ptr[1];
                    dst_ptr[2] = reg_ptr[2];
                    dst_ptr[3] = reg_ptr[3];
                }
            }
        }
    }
}

template <bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3, typename Engine4, typename Layout4,
          typename Engine5, typename Layout5>
__forceinline__ __device__ void copy_w_min_idx_kv(Tensor<Engine0, Layout0> const &S0,
                                                  Tensor<Engine1, Layout1> const &S1, Tensor<Engine2, Layout2> &D0,
                                                  Tensor<Engine3, Layout3> &D1, uint32_t *reg0, uint32_t *reg1,
                                                  Tensor<Engine4, Layout4> const &identity_MN,
                                                  Tensor<Engine5, Layout5> const &predicate_K, const int max_MN = 0,
                                                  const int min_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S0) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D0) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S0) == size<0>(D0));
    CUTE_STATIC_ASSERT_V(size<1>(S0) == size<1>(D0));
    CUTE_STATIC_ASSERT_V(size<2>(S0) == size<2>(D0));
    CUTE_STATIC_ASSERT_V(rank(S1) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D1) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S1) == size<0>(D1));
    CUTE_STATIC_ASSERT_V(size<1>(S1) == size<1>(D1));
    CUTE_STATIC_ASSERT_V(size<2>(S1) == size<2>(D1));
    CUTE_STATIC_ASSERT_V(size<0>(S0) == size<0>(S1));
    CUTE_STATIC_ASSERT_V(size<1>(S0) == size<1>(S1));
    CUTE_STATIC_ASSERT_V(size<2>(S0) == size<2>(S1));

#pragma unroll
    for (int m = 0; m < size<1>(S0); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S0); ++k) {
                const int idx = (m * size<2>(S0) + k) * 4;
                auto reg_ptr0 = reg0 + idx;
                auto reg_ptr1 = reg1 + idx;
                if (Is_even_K || predicate_K(k)) {
                    cute::copy_global_to_reg(S0(_, m, k), reg_ptr0);
                    cute::copy_global_to_reg(S1(_, m, k), reg_ptr1);
                }
            }
            flash::cp_async_wait<0>();
#pragma unroll
            for (int k = 0; k < size<2>(S0); ++k) {
                const int idx = (m * size<2>(S0) + k) * 4;
                auto reg_ptr0 = reg0 + idx;
                auto reg_ptr1 = reg1 + idx;
                if (Is_even_K || predicate_K(k)) {
                    auto dst_ptr0 = reinterpret_cast<uint32_t *>(D0(_, m, k).data().ptr_);
                    auto dst_ptr1 = reinterpret_cast<uint32_t *>(D1(_, m, k).data().ptr_);

                    dst_ptr0[0] = reg_ptr0[0];
                    dst_ptr0[1] = reg_ptr0[1];
                    dst_ptr0[2] = reg_ptr0[2];
                    dst_ptr0[3] = reg_ptr0[3];

                    dst_ptr1[0] = reg_ptr1[0];
                    dst_ptr1[1] = reg_ptr1[1];
                    dst_ptr1[2] = reg_ptr1[2];
                    dst_ptr1[3] = reg_ptr1[3];
                }
            }
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_global_to_reg(Tensor<Engine0, Layout0> const &S, uint32_t *D_ptr,
                                                   Tensor<Engine1, Layout1> const &identity_MN,
                                                   Tensor<Engine2, Layout2> const &predicate_K, const int max_MN = 0) {

    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                const int idx = m * size<2>(S) * 4 + k * 4;
                if (Is_even_K || predicate_K(k)) {
                    cute::copy_global_to_reg(S(_, m, k), D_ptr + idx);
                } else if (Clear_OOB_K) {
                    D_ptr[idx] = 0;
                    D_ptr[idx + 1] = 0;
                    D_ptr[idx + 2] = 0;
                    D_ptr[idx + 3] = 0;
                }
            }
        } else if (Clear_OOB_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                const int idx = m * size<2>(S) * 4 + k * 4;
                D_ptr[idx] = 0;
                D_ptr[idx + 1] = 0;
                D_ptr[idx + 2] = 0;
                D_ptr[idx + 3] = 0;
            }
        }
    }
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_reg_to_share(uint32_t *S_ptr, Tensor<Engine0, Layout0> &D) {

#pragma unroll
    for (int m = 0; m < size<1>(D); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(D); ++k) {
            const int idx = m * size<2>(D) * 4 + k * 4;
            cute::copy_reg_to_share(S_ptr + idx, D(_, m, k));
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_global_to_reg_V(Tensor<Engine0, Layout0> const &S, uint32_t *D_ptr,
                                                     Tensor<Engine1, Layout1> const &identity_MN,
                                                     Tensor<Engine2, Layout2> const &predicate_K, const int offset,
                                                     int max_MN = 0) {

    if (Is_even_MN == false || Is_even_K == false) {
        copy_global_to_reg<Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(S, D_ptr, identity_MN, predicate_K,
                                                                             max_MN);
        return;
    }

    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));

    typedef __NATIVE_VECTOR__(4, int) VecType;
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            const int idx = m * size<2>(S) * 4 + k * 4;
            auto src_ptr = (VecType *)(reinterpret_cast<uint32_t *>(S(_, m, k).data().ptr_) + offset);
            auto dst_ptr = (VecType *)(D_ptr + idx);
            dst_ptr[0] = __builtin_mxc_load_global_async128(src_ptr);
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_reg_to_share_V(uint32_t *S_ptr, Tensor<Engine0, Layout0> &D, const int offset) {

    if (Is_even_MN == false || Is_even_K == false) {
        copy_reg_to_share(S_ptr, D);
        return;
    }

    typedef __NATIVE_VECTOR__(4, int) VecType;

    cute::reg_trans(S_ptr[0], S_ptr[1]);
    cute::reg_trans(S_ptr[2], S_ptr[3]);
#pragma unroll
    for (int m = 0; m < size<1>(D); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(D); ++k) {
            const int idx = (m * size<2>(D) + k);
            auto D_ptr_Vec = (VecType *)(reinterpret_cast<uint32_t *>(D(_, m, k).data().ptr_) + offset);
            auto S_ptr_Vec = (VecType *)(S_ptr + 4 * idx);
            D_ptr_Vec[0] = S_ptr_Vec[0];
            if ((idx + 1) < size<1>(D) * size<2>(D)) {
                cute::reg_trans(S_ptr[4 * (idx + 1)], S_ptr[4 * (idx + 1) + 1]);
                cute::reg_trans(S_ptr[4 * (idx + 1) + 2], S_ptr[4 * (idx + 1) + 3]);
            }
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                        TiledMma tiled_mma, uint32_t *cpy_offset, const uint32_t &tCsB_stride,
                                        const uint32_t &tCrB_stride) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    cute::copy_trans(tCsB(_, _, _0{}), tCrB(_, _, _0{}), tCsB_stride, tCrB_stride, cpy_offset);

#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (Is_even_MN == false || Is_even_K == false) {
            cute::tensor_trans(tCrB(_, _, i), tCrB_stride);
        }
        if (i < size<2>(tCrA) - 1) {
            cute::copy_trans(tCsB(_, _, i + 1), tCrB(_, _, i + 1), tCsB_stride, tCrB_stride, cpy_offset);
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void swap_fragment(Tensor<Engine, Layout> &S) {
    using data_type = typename Engine::value_type;
    static_assert(decltype(size<0>(S))::value == 8);
    static_assert(std::is_same_v<data_type, mctlass::half_t> || std::is_same_v<data_type, mctlass::bfloat16_t>);

#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
#pragma unroll
        for (int n = 0; n < size<2>(S); ++n) {
            uint64_t *first = reinterpret_cast<uint64_t *>(S(_, m, n).data());
            uint64_t *second = first + 1;
            uint64_t tmp = *first;
            *first = *second;
            *second = tmp;
        }
    }
}

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void permute_4x4_b16(Tensor<Engine, Layout> &t) {
    using data_type = typename Engine::value_type;
    Tensor tPerm = make_tensor<data_type>(Shape<_4, _4>{});
    uint32_t v1, v2;
    uint32_t *dest;

#pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        cute::copy(tPerm, t(_, _, i));
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void stmatrix_trans(Tensor<Engine, Layout> &tCrC, void *sC) {
    using Dtype = typename Engine::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == _4{});
    CUTE_STATIC_ASSERT_V(size<2>(tCrC) == _2{});

#pragma unroll
    for (int i = 0; i < size<2>(tCrC); ++i) {
        auto r = reinterpret_cast<uint32_t *>(tCrC(_, 0, i).data().get());
        cute::reg_trans(r[0], r[1]);
    }

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);
    constexpr int SmemThreadsPerRow = SmemSizePerRow / SmemElemsPerLoad;

    auto row = __lane_id() / SmemThreadsPerRow;
    auto col = __lane_id() % SmemThreadsPerRow;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sC) + threadIdx.x / 64 * (SmemElemsPerLoad * 64) +
                      row * SmemSizePerRow + SmemElemsPerLoad * col;
    Tensor tCsC = make_tensor(make_smem_ptr(smem_ptr),
                              make_layout(Shape<_4, _1, _2>{}, Stride<_1, _0, Int<16 * SmemSizePerRow>>{}));
    cute::copy(tCrC, tCsC);
}

template <typename Engine, typename TLayout>
__forceinline__ __device__ void ldmatrix_trans(Tensor<Engine, TLayout> &tArA, void *sA) {
    using Dtype = typename Engine::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);
    CUTE_STATIC_ASSERT_V(size<0>(tArA) == _4{});

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);

    using LdsLayoutAtom = decltype(tile_to_shape(
        Layout<Shape<_4, Int<ThreadsPerGroup>>, Stride<Int<ThreadsPerGroup>, _1>>{}, Shape<_4, _16>{}));
    LdsLayoutAtom layout_atom;
    auto coord = layout_atom.get_hier_coord(__lane_id());
    auto row = cute::get<0>(coord);
    auto col_coord = cute::get<1>(coord);
    auto col = cute::get<0>(col_coord) + cute::get<1>(col_coord) * ThreadsPerGroup;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sA) + threadIdx.x / 64 % 2 * (8 * SmemSizePerRow) +
                      row * SmemSizePerRow + SmemElemsPerLoad * col;
    Tensor tAsA =
        make_tensor(make_smem_ptr(smem_ptr),
                    make_layout(Shape<_4, _2, _2>{}, Stride<_1, Int<16 * SmemSizePerRow>, Int<4 * SmemSizePerRow>>{}));
    cute::copy(tAsA, tArA);
}

#define SWIZZLE_STORE_QDO(smem_s, reg, smem_d) \
    cute::copy(smem_s, reg);                   \
    if (tidx / 8 % 2 == 1) {                   \
        flash::swap_fragment(reg);             \
    }                                          \
    cute::copy(reg, smem_d);

__forceinline__ __device__ void copy_share_reg_trans(uint64_t smem_ptr, uint32_t *rmem_ptr, uint32_t *cpy_offset,
                                                     const int reg_size) {
    smem_ptr = smem_ptr - cpy_offset[4];
    uint32_t __attribute__((address_space(3))) * src_ptr[4];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        src_ptr[i] = (uint32_t __attribute__((address_space(3))) *)(smem_ptr) + cpy_offset[i];
    }
    CUTE_UNROLL
    for (int i = 0; i < reg_size / 4; ++i) {
        rmem_ptr[2 * i] = src_ptr[i][0];
        rmem_ptr[2 * i + 1] = src_ptr[i][1];
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_rs_hdim64(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                               TiledMma tiled_mma, uint32_t *cpy_offset, const uint32_t tCrB_stride) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    auto rmem_ptr = reinterpret_cast<uint32_t *>(tCrB(_, _, _0{}).data());
    const int reg_size = size(tCrB(_, _, _0{}));
    copy_share_reg_trans(reinterpret_cast<uint64_t const>(tCsB(_, _, _0{}).data().ptr_), rmem_ptr, cpy_offset,
                         reg_size);

#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (Is_even_MN == false || Is_even_K == false) {
            cute::tensor_trans(tCrB(_, _, i), tCrB_stride);
        }
        if (i < size<2>(tCrA) - 1) {
            auto rmem_ptr = reinterpret_cast<uint32_t *>(tCrB(_, _, i + 1).data());
            copy_share_reg_trans(reinterpret_cast<uint64_t const>(tCsB(_, _, i + 1).data().ptr_), rmem_ptr, cpy_offset,
                                 reg_size);
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

}  // namespace flash
