cmake_minimum_required( VERSION 3.16 )

project(mcflashAttn VERSION 1.0)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

set(CMAKE_C_COMPILER ${MACA_PATH}/tools/cu-bridge/bin/cucc)
set(CMAKE_CXX_COMPILER ${MACA_PATH}/tools/cu-bridge/bin/cucc)

set(sources
    csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim224_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim224_bf16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu
)

function(cp_rename_file target)
    foreach(arg ${ARGN})
        get_filename_component(FILE_ABSOLUTE_NAME ${arg} ABSOLUTE)
        get_filename_component(FILE_NAME ${arg} NAME)
        string(REGEX REPLACE "sm80.cu" "sm80.cpp" NEW_FILE_NAME ${FILE_NAME})
        configure_file(${FILE_ABSOLUTE_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${NEW_FILE_NAME} COPYONLY)
    endforeach()
endfunction()

cp_rename_file(mcFlashAttn_cpy ${sources})

set(build_sources
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim32_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim32_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim64_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim64_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim96_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim96_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim128_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim128_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim160_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim160_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim192_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim192_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim224_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim224_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim256_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_hdim256_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim32_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim32_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim64_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim64_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim96_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim96_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim128_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim128_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim160_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim160_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim192_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim192_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim224_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim224_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim256_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_fwd_split_hdim256_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_bwd_hdim64_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_bwd_hdim64_bf16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_bwd_hdim128_fp16_sm80.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/flash_bwd_hdim128_bf16_sm80.cpp
)


add_compile_options(-x maca)
add_compile_options(--compiler-options -fPIC)
add_compile_options(-O3)
add_compile_options(-std=c++17)
add_compile_options(--expt-relaxed-constexpr)
add_compile_options(--expt-extended-lambda)
add_compile_options(--use_fast_math)
add_compile_options("SHELL:-mllvm -metaxgpu-inlinescope=50")
add_compile_options("SHELL:-mllvm -metaxgpu-disable-bsm-offset=0")
add_compile_options(-Xclang -menable-no-nans)
add_compile_options("SHELL:-gencode arch=compute_80,code=sm_80")
add_compile_options("SHELL:-gencode arch=compute_90,code=sm_90")
add_compile_options("SHELL:-mllvm -metaxgpu-enable-promote-kernel-arguments")
add_compile_options("SHELL:-mllvm -metaxgpu-const-KArgs-LDU=1")
add_compile_options("SHELL:-mllvm -metaxgpu-enable-unorder-dispatch")

add_compile_definitions(EXPORT_LIB)
add_compile_definitions(USE_MACA)
add_compile_definitions(NV_ARCH_A100)
add_compile_definitions(__FAST_HALF_CVT__)
add_compile_definitions(__MERGE_LDS_B64)

add_library(mcFlashAttn SHARED ${build_sources})
set_target_properties(mcFlashAttn PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(mcFlashAttn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${MACA_PATH}/tools/cu-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mctlass/include
)

set_target_properties(mcFlashAttn PROPERTIES
                    INSTALL_RPATH "$ORIGIN;$ORIGIN/../../lib;$ORIGIN/../../lib64;$ORIGIN/../lib64"
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_attn.h DESTINATION include/flash_attn)
install(TARGETS mcFlashAttn LIBRARY DESTINATION lib)
# set cpack using param
set( LIBRARY_NAME mcflashattn )
include (cmake/CPack.cmake)
