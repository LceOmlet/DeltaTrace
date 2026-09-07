// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"
#ifdef EXPORT_LIB
namespace mcFlashAttn {
    template <>
    void run_mha_fwd_splitkv_dispatch<mctlass::bfloat16_t, 160>(Flash_fwd_params &params, cudaStream_t stream) {
        run_mha_fwd_splitkv_hdim160<mctlass::bfloat16_t>(params, stream);
    }
}
#else
template void run_mha_fwd_splitkv_dispatch<mctlass::bfloat16_t, 160>(Flash_fwd_params &params, cudaStream_t stream);
#endif