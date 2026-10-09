// v4_coarsened: single change vs v2: thread coarsening (2x2 outputs per thread).
// Hypothesis (LOG.md 2026-10-09): v2/v3 run 32x32 = 1024 threads per block, the
// architectural maximum, so most SMs co-reside a single block and each thread
// does one output (no register reuse across outputs). v4 keeps the same 32x32
// output tile and shared-memory algorithm but uses 16x16 = 256 threads, each
// accumulating a 2x2 micro-tile in registers: 4x the work per thread, A-row and
// B-column elements reused across the micro-tile, and 4x more blocks eligible
// for concurrent residency (better latency hiding under WDDM scheduling noise).
// If v4 does not beat v2, the hypothesis fails and the numbers go to
// docs/NEGATIVE_RESULTS.md (occupancy was not the binding constraint).
// Bounds: same predicated zero-fill tile loads as v2 plus 4 guarded stores.
// Adapted idea note: thread coarsening per CUDA C++ Best Practices Guide;
// implementation written from scratch, see docs/REFERENCES.md.
#include "kern/cuda_check.h"
#include "kern/kernels.h"

#include <cstddef>

namespace kern {
namespace {

constexpr int TILE = 32;  // output tile edge (same as v2)
constexpr int BLOCK = 16; // threads per block edge (256 threads total)
constexpr int COARSE = 2; // outputs per thread edge (2x2 micro-tile)

__global__ void
v4_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M,
          int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  // This thread's 2x2 outputs: rows baseRow..baseRow+1, cols baseCol..baseCol+1.
  int baseRow = blockIdx.y * TILE + threadIdx.y * COARSE;
  int baseCol = blockIdx.x * TILE + threadIdx.x * COARSE;
  float acc[COARSE][COARSE] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  int numTiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < numTiles; ++t) {
    // Cooperative tile load: 256 threads x 4 elements each = full 32x32 tile.
    // Element (r, c) of the tile is loaded by thread (r/2... ) via linear id.
    int tid = threadIdx.y * BLOCK + threadIdx.x; // 0..255
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int elem = tid * 4 + e;
      int lr = elem / TILE, lc = elem % TILE; // local tile coords
      int aCol = t * TILE + lc;
      int bRow = t * TILE + lr;
      int gRow = blockIdx.y * TILE + lr;
      int gCol = blockIdx.x * TILE + lc;
      As[lr][lc] = (gRow < M && aCol < K) ? A[(std::size_t)gRow * K + aCol] : 0.0f;
      Bs[lr][lc] = (bRow < K && gCol < N) ? B[(std::size_t)bRow * N + gCol] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
#pragma unroll
      for (int i = 0; i < COARSE; ++i) {
#pragma unroll
        for (int j = 0; j < COARSE; ++j) {
          acc[i][j] += As[threadIdx.y * COARSE + i][k] * Bs[k][threadIdx.x * COARSE + j];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < COARSE; ++i) {
#pragma unroll
    for (int j = 0; j < COARSE; ++j) {
      int r = baseRow + i, c = baseCol + j;
      if (r < M && c < N)
        C[(std::size_t)r * N + c] = acc[i][j];
    }
  }
}

} // namespace

void
launch_v4_coarsened(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(BLOCK, BLOCK);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  v4_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace kern
