// v0_naive: baseline. One thread per C element, dot-product over K.
// Hypothesis (LOG.md step 0): correctness first; performance expected to be poor
// because B is read column-wise (stride N) giving uncoalesced loads, and every
// C element re-reads its full row of A and column of B (no data reuse).
// Handles arbitrary M/N/K via bounds check.
// Adapted idea note: standard textbook SGEMM; no code copied from references
// (see docs/REFERENCES.md: CUDA C++ Programming Guide, matrixMul sample).
#include "kern/cuda_check.h"
#include "kern/kernels.h"

namespace kern {
namespace {

__global__ void v0_kernel(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
#pragma unroll 1  // v0: deliberately no unrolling; v3 tests unroll/vectorization.
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}

}  // namespace

void launch_v0_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  v0_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
}

}  // namespace kern
