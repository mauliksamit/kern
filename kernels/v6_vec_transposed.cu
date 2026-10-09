// v6_vec_transposed: single change vs v5: float4 global tile loads plus the A
// tile stored TRANSPOSED in shared memory (At[8][128] instead of As[128][8]).
// Hypothesis (LOG.md Phase 2): shared-memory load instructions per FMA drop.
// Two mechanisms: (a) one float4 per thread replaces 4 scalar loads in the
// cooperative fill (fewer global transactions on aligned full tiles);
// (b) v5's A-fragment pattern As[ty*8+i][k] puts the two ty-groups of a warp
// on the SAME bank at different addresses (rows 8 apart alias: 8 floats/row
// is a 32-bank period divisor) — a 2-way conflict on every A-fragment load —
// while transposed At[k][ty*8+i] gives one address per ty-group (broadcast,
// no conflict). Expected: small but measurable gain at large K, where the
// compute loop dominates; if within noise, v6 is a recorded negative result
// (layout second-order, like v3's loads).
// Correctness: fast path needs fullRow && fullCol && fullK-slice && K % 4 == 0
// && N % 4 == 0 && aligned tile bases (so every row shares 16B alignment);
// otherwise the scalar fallback fills the SAME transposed layout. The decision
// is block-uniform (tile-level terms only) and both branches share one barrier
// structure — never a repeat of the v3 divergent-barrier bug. Arbitrary M/N/K
// via predicated fills + guarded stores; size_t index math throughout.
// Adapted idea note: transposed/shared-swizzled operand storage per the public
// SGEMM-optimization write-ups (docs/REFERENCES.md #8); written from scratch.
#include "kern/cuda_check.h"
#include "kern/kernels.h"

#include <cstddef>
#include <cstdint>

namespace kern {
namespace {

constexpr int BM = 128;  // block-tile rows of C (same as v5)
constexpr int BN = 128;  // block-tile cols of C (same as v5)
constexpr int BK = 8;    // K-slice per step (same as v5)
constexpr int TM = 8;    // per-thread rows (same as v5)
constexpr int TN = 8;    // per-thread cols (same as v5)
constexpr int BDIM = 16; // 16x16 = 256 threads per block (same as v5)

__global__ void
v6_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M,
          int N, int K) {
  __shared__ float At[BK][BM]; // A tile TRANSPOSED: At[k][r]
  __shared__ float Bs[BK][BN]; // B tile as in v5

  const int baseRow = blockIdx.y * BM + threadIdx.y * TM;
  const int baseCol = blockIdx.x * BN + threadIdx.x * TN;
  float acc[TM][TN] = {{0.0f}};
  float aFrag[TM];
  float bFrag[TN];

  const int tid = threadIdx.y * BDIM + threadIdx.x; // 0..255
  const int numTiles = (K + BK - 1) / BK;
  for (int t = 0; t < numTiles; ++t) {
    const int tileK = t * BK;
    // Block-uniform fast-path decision (tile-level terms only).
    const bool fullRow = (blockIdx.y * BM + BM) <= M;
    const bool fullCol = (blockIdx.x * BN + BN) <= N;
    const bool fullK = (tileK + BK) <= K;
    const std::size_t aBase = (std::size_t)(blockIdx.y * BM) * (std::size_t)K + (std::size_t)tileK;
    const std::size_t bBase = (std::size_t)tileK * (std::size_t)N + (std::size_t)(blockIdx.x * BN);
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         ((reinterpret_cast<uintptr_t>(A + aBase) % 16) == 0) &&
                         ((reinterpret_cast<uintptr_t>(B + bBase) % 16) == 0);
    if (fullRow && fullCol && fullK && aligned) {
      // Vectorized fill: 256 threads x float4 = 1024 floats = one tile.
      // A: tid -> ar = tid/2, ac = (tid%2)*4; B: br = tid/32, bc = (tid%32)*4.
      const int ar = tid / 2, ac = (tid % 2) * 4;
      const int br = tid / 32, bc = (tid % 32) * 4;
      const float4 a4 = reinterpret_cast<const float4*>(
          A + (std::size_t)(blockIdx.y * BM + ar) * (std::size_t)K + tileK + ac)[0];
      const float4 b4 = reinterpret_cast<const float4*>(
          B + (std::size_t)(tileK + br) * (std::size_t)N + blockIdx.x * BN + bc)[0];
      At[ac][ar] = a4.x;
      At[ac + 1][ar] = a4.y;
      At[ac + 2][ar] = a4.z;
      At[ac + 3][ar] = a4.w;
      Bs[br][bc] = b4.x;
      Bs[br][bc + 1] = b4.y;
      Bs[br][bc + 2] = b4.z;
      Bs[br][bc + 3] = b4.w;
    } else {
      // Scalar fallback into the SAME transposed layout (edge/misaligned tiles).
      for (int e = 0; e < 4; ++e) {
        const int elemA = tid * 4 + e;
        const int ar = elemA / BK, ac = elemA % BK;
        const int gRow = blockIdx.y * BM + ar;
        const int aCol = tileK + ac;
        At[ac][ar] = (gRow < M && aCol < K) ? A[(std::size_t)gRow * (std::size_t)K + aCol] : 0.0f;
        const int elemB = tid * 4 + e;
        const int br = elemB / BN, bc = elemB % BN;
        const int bRow = tileK + br;
        const int gCol = blockIdx.x * BN + bc;
        Bs[br][bc] = (bRow < K && gCol < N) ? B[(std::size_t)bRow * (std::size_t)N + gCol] : 0.0f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
        aFrag[i] = At[k][threadIdx.y * TM + i];
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
launch_v6_vec_transposed(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(BDIM, BDIM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  v6_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace kern
