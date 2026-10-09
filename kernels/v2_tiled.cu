// v2_tiled: single change vs v1: 32x32 shared-memory tiling.
// Hypothesis: v0/v1 re-read A and B O(N) times from DRAM. Tiling loads each
// TILE x TILE patch once into shared memory, cutting global traffic by ~TILE
// and making B reads shared (broadcast) instead of strided DRAM.
// TILE=32 matches the warp size: one tile row per warp -> no bank conflicts on
// the common path. Handles non-multiples via bounds-checked loads (zero-fill)
// and guarded stores.
// Adapted idea note: tiling strategy per "Tiled Matrix Multiplication" (UIUC /
// CUDA samples); written from scratch, see docs/REFERENCES.md.
#include "kern/cuda_check.h"
#include "kern/kernels.h"

namespace kern {
namespace {

constexpr int TILE = 32;

__global__ void v2_kernel(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  int numTiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < numTiles; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

}  // namespace

void launch_v2_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(TILE, TILE);
  // NOTE: 32x32 = 1024 threads = architectural max per block; portable but
  // occupancy-limited. v3 keeps TILE but adds vectorization instead of bigger tiles.
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  v2_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
}

}  // namespace kern
