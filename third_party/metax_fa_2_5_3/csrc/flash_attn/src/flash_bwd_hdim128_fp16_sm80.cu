// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"
#ifdef EXPORT_LIB
namespace mcFlashAttn {
#endif
template<>
void run_mha_bwd_<mctlass::half_t, 128>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<mctlass::half_t>(params, stream);
}
#ifdef EXPORT_LIB
}
#endif
