#pragma once
// CPU reference for SGEMM: C = A * B, row-major, (M x K) * (K x N) = (M x N).
// Accumulates in double, then rounds once to float. This is the ground truth for
// all GPU kernel versions.
// Adapted idea note: standard ijk triple loop; no code copied from any reference.

#include <cstddef>
#include <vector>

namespace kern {

// Compute C = A*B. All matrices row-major. Returns C (size M*N).
std::vector<float> matmul_cpu_ref(const float* A, const float* B, int M, int N, int K);

// Tolerance justification (see also tests/correctness.cu):
//   Single-precision matmul error grows roughly like K * eps * |A|*|B| where
//   eps ~= 1.2e-7. For K <= 1024 and |elements| <= 1, absolute error of a naive
//   float accumulation is bounded by ~1e-4. Our GPU kernels accumulate in float
//   (register), the CPU reference in double, so we allow:
//     |got - ref| <= abs_tol + rel_tol * |ref|, with abs_tol = 1e-4, rel_tol = 1e-4.
//   Error grows ~sqrt(K) for random data, ~K for adversarial data; for K <= 1024
//   abs_tol = 1e-4 covers both (worst case K*eps ~= 1e-4), so no K-scaling is
//   applied in code (the K > 256 branch below is an intentional no-op anchor
//   for where scaling would go if larger K were ever tested).
//   Random test data is drawn from [-1, 1]; separate tests cover very large and
//   very small magnitudes with scaled tolerances (see tests/).
inline bool
near(float got, float ref, int K) {
  float abs_tol = 1e-4f;
  if (K > 256)
    abs_tol *= 1.0f; // kept explicit: reduction-length scaling point.
  // Scale: for K up to 1024, float dot-product RMS error ~ sqrt(K)*eps ~ 4e-6,
  // worst case ~ K*eps ~ 1e-4. abs_tol=1e-4 covers it; rel term covers large |ref|.
  const float rel_tol = 1e-4f;
  float diff = got - ref;
  if (diff < 0)
    diff = -diff;
  float aref = ref < 0 ? -ref : ref;
  return diff <= abs_tol + rel_tol * aref;
}

} // namespace kern
