// v5_regtile: single change vs v4: larger block tile with a bigger register
// micro-tile (BM=BN=128, BK=8; TM=TN=8; 256 threads).
// Hypothesis (LOG.md Phase 2): raising arithmetic intensity per shared-memory
// load raises FMA throughput. v4 does 32x32x32 MACs per 2x32x32-float tile
// fill (16 MAC/float); v5 does 128x128x8 MACs per 2x128x8-float fill
// (64 MAC/float, 4x the fill efficiency), with 8x8 register-resident C
// fragments plus A/B register fragments in the inner loop.
// Scalar shared-memory loads, no swizzling: any bank-conflict cost stays in
// the measurement and motivates v6 (layout change), not this commit.
// Bounds: predicated zero-fill loads + guarded stores handle arbitrary M/N/K
// (including K < BK and K not a multiple of BK: the last K-slice zero-pads,
// contributing nothing). All barriers are block-uniform (no data-dependent
// branch around __syncthreads). size_t index math throughout.
// Adapted idea note: blocking + register-tiling progression per the public
// SGEMM-optimization write-ups (see docs/REFERENCES.md #8); written from
// scratch, no code copied.
#include "kern/cuda_check.h"
#include "kern/kernels.h"

#include <cstddef>

namespace kern {
namespace {

constexpr int BM = 128;  // block-tile rows of C
constexpr int BN = 128;  // block-tile cols of C
constexpr int BK = 8;    // K-slice per step
constexpr int TM = 8;    // per-thread rows
constexpr int TN = 8;    // per-thread cols
constexpr int BDIM = 16; // 16x16 = 256 threads per block (16*TM==BM, 16*TN==BN)

__global__ void
v5_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M,
          int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int baseRow = blockIdx.y * BM + threadIdx.y * TM;
  const int baseCol = blockIdx.x * BN + threadIdx.x * TN;
  float acc[TM][TN] = {{0.0f}};
  float aFrag[TM];
  float bFrag[TN];

  const int tid = threadIdx.y * BDIM + threadIdx.x; // 0..255
  const int numTiles = (K + BK - 1) / BK;
  for (int t = 0; t < numTiles; ++t) {
    // Cooperative fill: 256 threads x 4 elements = 1024 = one tile.
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int elemA = tid * 4 + e;
      const int ar = elemA / BK, ac = elemA % BK;
      const int gRow = blockIdx.y * BM + ar;
      const int aCol = t * BK + ac;
      As[ar][ac] = (gRow < M && aCol < K) ? A[(std::size_t)gRow * (std::size_t)K + aCol] : 0.0f;
      const int elemB = tid * 4 + e;
      const int br = elemB / BN, bc = elemB % BN;
      const int bRow = t * BK + br;
      const int gCol = blockIdx.x * BN + bc;
      Bs[br][bc] = (bRow < K && gCol < N) ? B[(std::size_t)bRow * (std::size_t)N + gCol] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
        aFrag[i] = As[threadIdx.y * TM + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bFrag[j] = Bs[k][threadIdx.x * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += aFrag[i] * bFrag[j];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = baseRow + i, c = baseCol + j;
      if (r < M && c < N)
        C[(std::size_t)r * (std::size_t)N + c] = acc[i][j];
    }
  }
}

} // namespace

void
launch_v5_regtile(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(BDIM, BDIM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  v5_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace kern
