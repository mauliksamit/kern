cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# kern: SGEMM optimization ladder (v0 naive -> v5 register-tiled) + cuBLAS baseline.
# One command to build, one to test:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# GPU targets are built only when a CUDA toolkit is found. CPU reference,
# CPU tests, and audit/table scripts always work without a GPU (CI-safe).
# ---------------------------------------------------------------------------
project(kern LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release flags (never time a -G debug build). -lineinfo is added to CUDA
# release flags below so Nsight Compute can map back to source lines.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Try to find CUDA quietly: CPU-only build must still work (CI has no GPU).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  set(KERN_HAVE_CUDA ON)
else()
  set(KERN_HAVE_CUDA OFF)
  message(STATUS "[kern] No CUDA compiler found -- building CPU-only targets. "
                 "Install the CUDA toolkit (12.8+ for RTX 50-series, sm_120) to build GPU targets.")
endif()

if(KERN_HAVE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Detectable arch default: RTX 5070 (Blackwell GB205) is sm_120 and needs
  # CUDA 12.8+. Override with -DKERN_CUDA_ARCH=sm_XX.
  # NOTE: this default is an inference from nvidia-smi (no nvcc here to confirm);
  # run tools/detect_arch.py after installing the toolkit and set the flag explicitly.
  if(NOT DEFINED KERN_CUDA_ARCH)
    set(KERN_CUDA_ARCH "sm_120")
  endif()
  # Convert sm_120 -> 120 for CMAKE_CUDA_ARCHITECTURES.
  string(REGEX MATCH "[0-9]+" KERN_CUDA_ARCH_NUM "${KERN_CUDA_ARCH}")
  set(CMAKE_CUDA_ARCHITECTURES ${KERN_CUDA_ARCH_NUM})
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo --use_fast_math -Xcompiler=/O2")
  message(STATUS "[kern] CUDA arch: ${KERN_CUDA_ARCH} (CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")
  # Bake the exact release flags into bench JSON (Phase 0 evidence rule: record
  # what actually built the binary; never guess). Stringified in bench.cu.
  add_compile_definitions(
    "KERN_COMPILE_FLAGS=-O3 -lineinfo --use_fast_math -Xcompiler=/O2 -arch=${KERN_CUDA_ARCH}")
endif()

# ---- CPU reference library (always built) ----------------------------------
add_library(kern_cpu src/matmul_cpu.cpp)
target_include_directories(kern_cpu PUBLIC include)
target_compile_options(kern_cpu PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wextra>)

# ---- GPU library (only with toolkit) ---------------------------------------
if(KERN_HAVE_CUDA)
  add_library(kern_kernels STATIC
    kernels/v0_naive.cu
    kernels/v1_coalesced.cu
    kernels/v2_tiled.cu
    kernels/v3_tiled_vectorized.cu
    kernels/v4_coarsened.cu
    kernels/v5_regtile.cu
  )
  target_include_directories(kern_kernels PUBLIC include)
  set_target_properties(kern_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  if(CUDAToolkit_FOUND)
    target_link_libraries(kern_kernels PUBLIC CUDA::cublas CUDA::cudart)
    target_compile_definitions(kern_kernels PUBLIC KERN_HAS_CUBLAS)
  endif()

  add_executable(kern_bench src/bench.cu)
  target_link_libraries(kern_bench PRIVATE kern_kernels kern_cpu)
  if(CUDAToolkit_FOUND)
    # CUDA::cuda_driver for cuDeviceGetAttribute clocks (CUDA 13 dropped
    # clockRate from cudaDeviceProp); cublas = vendor baseline.
    target_link_libraries(kern_bench PRIVATE CUDA::cublas CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(kern_bench PRIVATE KERN_HAS_CUBLAS)
  endif()

  add_executable(kern_correctness tests/correctness.cu)
  target_link_libraries(kern_correctness PRIVATE kern_kernels kern_cpu)
  if(CUDAToolkit_FOUND)
    target_link_libraries(kern_correctness PRIVATE CUDA::cudart CUDA::cublas)
    target_compile_definitions(kern_correctness PRIVATE KERN_HAS_CUBLAS)
  endif()
endif()

# ---- Tests ------------------------------------------------------------------
include(CTest)
include(CTestUseLaunchers OPTIONAL)

# CPU-side checks always run (CI-safe, no GPU needed).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  add_test(NAME python_cpu_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_correctness_cpu.py)
endif()

if(KERN_HAVE_CUDA)
  # GPU correctness gate: many shapes incl. awkward ones, fixed seeds.
  add_test(NAME gpu_correctness COMMAND kern_correctness --seed 1234)
endif()

# Claims audit: every README number must trace to results/. Always runs.
if(Python3_FOUND)
  add_test(NAME claims_audit
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/claims_audit.py)
endif()
