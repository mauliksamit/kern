// v1_coalesced: single change vs v0: flattened 1D thread mapping + __restrict__ +
// contiguous row-major traversal so global loads/stores coalesce into 128B segments.
// Hypothesis: v0's 2D (16x16) blocks already coalesce reasonably, but flattened 1D
// with BLOCK=256 gives the hardware one fully coalesced row of C per warp and
// helps the compiler generate wider vectorized accesses for A row reads.
// Arbitrary sizes handled via linear bounds check.
// Adapted idea note: coalescing rules per CUDA C++ Programming Guide (see
// docs/REFERENCES.md); implementation written from scratch.
#include "kern/cuda_check.h"
#include "kern/kernels.h"

namespace kern {
namespace {

constexpr int V1_BLOCK = 256;

__global__ void v1_kernel(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (tid >= total) return;
  int row = tid / N;
  int col = tid % N;
  float acc = 0.0f;
  // A row segment A[row*K .. row*K+K) is contiguous -> coalesced across warp
  // only partially (each thread reads its own row), but C store is fully coalesced.
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[tid] = acc;
}

}  // namespace

void launch_v1_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
  int total = M * N;
  dim3 block(V1_BLOCK);
  dim3 grid((total + V1_BLOCK - 1) / V1_BLOCK);
  v1_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
}

}  // namespace kern
