// v3_tiled_vectorized: single change vs v2: float4-vectorized global loads +
// __launch_bounds__ + #pragma unroll for the K-loop.
// Hypothesis: v2's scalar shared-memory fill is DRAM-latency bound for large K;
// 128-bit vectorized loads halve the number of global transactions on aligned
// full tiles, and launch bounds let the compiler trade registers for occupancy.
// Correctness: fast path requires the full TILE x TILE output tile to be inside
// (M, N), the K-tile fully inside K, K % 4 == 0 and N % 4 == 0 (so every row in
// the tile shares the base pointer's 16B alignment), and aligned tile-base
// pointers; otherwise falls back to the scalar v2 path for that tile.
// The decision MUST be block-uniform: it is computed from tile-base pointers,
// never from per-thread (row, col). Threads deciding differently would execute
// different __syncthreads() barriers, which is undefined behavior and can hang
// (fixed 2026-10-09: the original used per-thread row/col alignment, which
// diverges whenever K % 4 != 0 or N % 4 != 0). Arbitrary M/N/K stay correct.
// NOTE (fixed 2026-10-03): first cut used __launch_bounds__(1024, 2); ptxas
// rejected the min-blocks hint (2 x 1024 threads exceeds SM residency), so it
// is (1024, 1). See LOG.md.
// Adapted idea note: vectorization guidance per CUDA C++ Best Practices Guide;
// implementation written from scratch, see docs/REFERENCES.md.
#include "kern/cuda_check.h"
#include "kern/kernels.h"

#include <cstdint>

namespace kern {
namespace {

constexpr int TILE = 32;

__global__ void
__launch_bounds__(1024, 1) v3_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  int numTiles = (K + TILE - 1) / TILE;
  int tid = threadIdx.y * TILE + threadIdx.x; // 0..1023 linear id in block
  // Each thread covers 8 floats in the vectorized path (32x32/128 threads...).
  // Simpler model: threads with tid < 256 cooperatively load As/Bs as float4.
  for (int t = 0; t < numTiles; ++t) {
    int tileK = t * TILE;
    bool fullK = (tileK + TILE) <= K;
    bool fullRow = (blockIdx.y * TILE + TILE) <= M;
    bool fullCol = (blockIdx.x * TILE + TILE) <= N;
    // Block-uniform fast-path decision (see header note): tile-base alignment
    // plus K % 4 == 0 / N % 4 == 0 so every row in the tile shares it.
    // A is float (4B): element index p is 16B-aligned iff p % 4 == 0.
    std::size_t aBase = (std::size_t)(blockIdx.y * TILE) * (std::size_t)K + (std::size_t)tileK;
    std::size_t bBase = (std::size_t)tileK * (std::size_t)N + (std::size_t)(blockIdx.x * TILE);
    bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                   ((reinterpret_cast<uintptr_t>(A + aBase) % 16) == 0) &&
                   ((reinterpret_cast<uintptr_t>(B + bBase) % 16) == 0);

    if (fullK && fullRow && fullCol && aligned) {
      // Vectorized cooperative load: 256 threads x float4 = 1024 floats = one 32x32 tile.
      if (tid < 256) {
        int base = tid * 4; // element offset inside tile
        int ar = base / TILE, ac = base % TILE;
        int br = base / TILE, bc = base % TILE;
        float4 a4 =
            reinterpret_cast<const float4*>(A + (blockIdx.y * TILE + ar) * K + tileK + ac)[0];
        float4 b4 =
            reinterpret_cast<const float4*>(B + (tileK + br) * N + blockIdx.x * TILE + bc)[0];
        As[ar][ac] = a4.x;
        As[ar][ac + 1] = a4.y;
        As[ar][ac + 2] = a4.z;
        As[ar][ac + 3] = a4.w;
        Bs[br][bc] = b4.x;
        Bs[br][bc + 1] = b4.y;
        Bs[br][bc + 2] = b4.z;
        Bs[br][bc + 3] = b4.w;
      }
      __syncthreads();
    } else {
      // Scalar fallback (identical to v2): correct for edges / misaligned.
      int aCol = tileK + threadIdx.x;
      int bRow = tileK + threadIdx.y;
      As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
      Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
      __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N)
    C[row * N + col] = acc;
}

} // namespace

void
launch_v3_tiled_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  v3_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace kern
