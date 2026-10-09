#pragma once
// One file per kernel version so the progression is reproducible.
// Each version exposes a host launcher with identical signature; bench/ and
// correctness binaries link all four and compare against the CPU reference.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace kern {

// C = A*B, all row-major. A: MxK, B: KxN, C: MxN.
void launch_v0_naive(const float* A, const float* B, float* C, int M, int N, int K);
void launch_v1_coalesced(const float* A, const float* B, float* C, int M, int N, int K);
void launch_v2_tiled(const float* A, const float* B, float* C, int M, int N, int K);
void launch_v3_tiled_vectorized(const float* A, const float* B, float* C, int M, int N, int K);
void launch_v4_coarsened(const float* A, const float* B, float* C, int M, int N, int K);
void launch_v5_regtile(const float* A, const float* B, float* C, int M, int N, int K);

} // namespace kern
