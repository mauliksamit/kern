// CPU reference: accumulate in double.
#include "kern/matmul_cpu.h"

namespace kern {

std::vector<float>
matmul_cpu_ref(const float* A, const float* B, int M, int N, int K) {
  std::vector<float> C(static_cast<std::size_t>(M) * static_cast<std::size_t>(N), 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0; // double accumulation: ground truth.
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
      }
      C[i * N + j] = static_cast<float>(acc);
    }
  }
  return C;
}

} // namespace kern
