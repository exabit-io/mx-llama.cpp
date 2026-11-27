if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -Xarch_device -O2 -Xarch_host -O0")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.3)
    message(FATAL_ERROR "At least ROCM/HIP V6.3 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh" "../ggml-cuda/q8_repack/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu" "../ggml-cuda/q8_repack/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

ggml_cuda_fattn_instances(${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda SRCS)
list(APPEND GGML_SOURCES_ROCM ${SRCS})

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (GGML_HIP_GFX906_MAX_ILP AND "gfx906" IN_LIST CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -mllvm -amdgpu-sched-strategy=max-ilp")
endif()

# LLVM 24 with max-ilp spills VGPRs in the inner loop of the head-256 tile kernels (16 in flash_attn_tile<256,256,16,2>, LLVM 23: 1): prefill -2.6% at 1x128K, -4.1% at 1x255K on Qwen3.8-27B Q8_0, four gfx906.
# A scheduler VGPR limit removes the spills. The option changes every kernel of the object and each batch size runs another one:
# 90% left 20 reloads per iteration in flash_attn_tile<256,256,4,2> (3 or 4 tokens per sequence: MTP verification of 3 drafted tokens,
# decode -1.8% at 118K depth). 85% left none in any of the 34 kernels of llama.cpp v1.0.0. The option does not exist before LLVM 24.
# llama.cpp v1.1.0 (upstream #32901 rewrote the kernel, 17 instances): 85% leaves 19 reloads per iteration in <8,1>, 90% two in <4,2>;
# 88% and 75% leave none anywhere. Timed on four MI50s, Qwen3.8-27B Q8_0, against the same build with this one object by LLVM 23:
# 88% is level at 4x64K and 1x255K (decode -0.1%, prefill +0.3..+0.5%) and +7.2% MTP decode at 118K depth; without a limit
# prefill is -2.4% / -5.4%; 75% costs decode -2.3% / -7.0% and changes the text of an MTP run.
if (GGML_HIP_GFX906_MAX_ILP AND "gfx906" IN_LIST CMAKE_HIP_ARCHITECTURES
        AND NOT CXX_IS_HIPCC AND CMAKE_HIP_COMPILER_VERSION VERSION_GREATER_EQUAL 24)
    # The instance is named by K/V type since llama.cpp v1.1.0 (fattn-tile-instance-f16-f16-dkq256-dv256.cu). A property on
    # a source that does not exist is silently ignored, so the name is looked up and a miss stops the configure.
    file(GLOB GGML_HIP_TILE256_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda/template-instances/fattn-tile-instance-*dkq256-dv256.cu")
    if (NOT GGML_HIP_TILE256_SRCS)
        message(FATAL_ERROR "gfx906 with LLVM 24: no head-256 tile instance found for the VGPR limit (was the instance renamed?)")
    endif()
    set_property(SOURCE ${GGML_HIP_TILE256_SRCS}
                 APPEND_STRING PROPERTY COMPILE_FLAGS " -mllvm -amdgpu-vgpr-threshold-percent=88")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)
