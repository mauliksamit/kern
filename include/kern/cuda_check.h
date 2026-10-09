#pragma once
// Every CUDA API call must be wrapped in CUDA_CHECK.
// Kernel launches must additionally be followed by cudaGetLastError.
// In debug builds callers synchronize (CUDA_SYNC_DEBUG) so launch failures surface early.

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                                          \
  do {                                                                                            \
    cudaError_t _kern_err = (expr);                                                               \
    if (_kern_err != cudaSuccess) {                                                               \
      std::fprintf(stderr, "[CUDA_CHECK] %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,     \
                   cudaGetErrorString(_kern_err));                                                \
      std::exit(EXIT_FAILURE);                                                                    \
    }                                                                                             \
  } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                                   \
  do {                                                                                            \
    cudaError_t _kern_err = cudaGetLastError();                                                   \
    if (_kern_err != cudaSuccess) {                                                               \
      std::fprintf(stderr, "[CUDA_CHECK] cudaGetLastError at %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(_kern_err));                                                \
      std::exit(EXIT_FAILURE);                                                                    \
    }                                                                                             \
  } while (0)

#ifdef NDEBUG
#define CUDA_SYNC_DEBUG() ((void)0)
#else
#define CUDA_SYNC_DEBUG()                                                                         \
  do {                                                                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  } while (0)
#endif

#else
// Non-CUDA translation units (CPU-only builds): no-ops so headers stay includable.
#define CUDA_CHECK(expr) ((void)0)
#define CUDA_CHECK_LAST_ERROR() ((void)0)
#define CUDA_SYNC_DEBUG() ((void)0)
#endif
