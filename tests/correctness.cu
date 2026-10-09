// Correctness gate.
// Compares each kernel version against the CPU double-accumulated reference on
// many shapes, including awkward ones (size 1, primes, non-multiples of 32,
// very large / very small magnitudes). Fixed seeds, logged in output + JSON.
// Tolerance: see kern/matmul_cpu.h `near()` -- rel 1e-4 + abs 1e-4, justified by
// K*eps error growth for float dot products (eps ~ 1.2e-7). Large-magnitude and
// small-magnitude cases use scaled checks documented inline.
// Usage: kern_correctness [--seed 1234] [--out results/correctness_<...>.json]
#include "kern/cuda_check.h"
#include "kern/kernels.h"
#include "kern/matmul_cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef KERN_HAS_CUBLAS
#include <cublas_v2.h>
#endif

namespace {

struct Shape {
  int M, N, K;
  const char* tag;
};

void
fill(std::mt19937& rng, std::vector<float>& v, float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  for (auto& x : v)
    x = d(rng);
}

int
check_one(const char* name, void (*launch)(const float*, const float*, float*, int, int, int),
          const float* dA, const float* dB, float* dC, const std::vector<float>& hA,
          const std::vector<float>& hB, int M, int N, int K, int seed) {
  launch(dA, dB, dC, M, N, K);
  CUDA_CHECK_LAST_ERROR();
  CUDA_SYNC_DEBUG(); // sync in debug builds so failures surface here.
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hC(static_cast<std::size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> ref = kern::matmul_cpu_ref(hA.data(), hB.data(), M, N, K);
  int bad = 0;
  float worst = 0.0f;
  for (std::size_t i = 0; i < hC.size(); ++i) {
    float diff = hC[i] - ref[i];
    if (diff < 0)
      diff = -diff;
    if (diff > worst)
      worst = diff;
    if (!kern::near(hC[i], ref[i], K)) {
      if (bad < 5) {
        std::printf("    mismatch %s [%d] got=%g ref=%g (M=%d N=%d K=%d seed=%d)\n", name, (int)i,
                    hC[i], ref[i], M, N, K, seed);
      }
      ++bad;
    }
  }
  std::printf("  %-18s M=%-5d N=%-5d K=%-5d worst_abs_err=%.3g %s\n", name, M, N, K, worst,
              bad == 0 ? "PASS" : "FAIL");
  return bad;
}

#ifdef KERN_HAS_CUBLAS
// Validate the vendor baseline computes the same math (Phase B).
int
check_cublas(cublasHandle_t h, const float* dA, const float* dB, float* dC,
             const std::vector<float>& hA, const std::vector<float>& hB, int M, int N, int K,
             int seed) {
  // Same derivation as src/bench.cu time_cublas: col-major C^T = B^T * A^T via
  // swapped buffers with CUBLAS_OP_N, valid for all shapes. dC holds C^T
  // (N x M row-major), so C[i][j] is read as hCt[j*M+i].
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st =
      cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::printf("  %-18s M=%-5d N=%-5d K=%-5d cublasSgemm FAILED\n", "cublas", M, N, K);
    return 1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hCt(static_cast<std::size_t>(N) * M); // C^T, N x M col-major
  CUDA_CHECK(cudaMemcpy(hCt.data(), dC, hCt.size() * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> ref = kern::matmul_cpu_ref(hA.data(), hB.data(), M, N, K);
  int bad = 0;
  float worst = 0.0f;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      // Col-major N x M: (C^T)[x][y] lives at y*N+x; C[i][j]=(C^T)[j][i] -> i*N+j.
      float got = hCt[(std::size_t)i * N + j];
      float diff = got - ref[(std::size_t)i * N + j];
      if (diff < 0)
        diff = -diff;
      if (diff > worst)
        worst = diff;
      if (!kern::near(got, ref[(std::size_t)i * N + j], K)) {
        if (bad < 5) {
          std::printf("    mismatch cublas [%d,%d] got=%g ref=%g (M=%d N=%d K=%d seed=%d)\n", i, j,
                      got, ref[(std::size_t)i * N + j], M, N, K, seed);
        }
        ++bad;
      }
    }
  }
  std::printf("  %-18s M=%-5d N=%-5d K=%-5d worst_abs_err=%.3g %s\n", "cublas", M, N, K, worst,
              bad == 0 ? "PASS" : "FAIL");
  return bad;
}
#endif

} // namespace

int
main(int argc, char** argv) {
  int seed = 1234;
  std::string out;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
      seed = std::atoi(argv[++i]);
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
      out = argv[++i];
  }
  std::printf("[kern_correctness] seed=%d\n", seed);

  // Awkward-shape coverage.
  const Shape shapes[] = {
      {1, 1, 1, "size-1"},
      {1, 32, 16, "row-vector"},
      {32, 1, 16, "col-vector"},
      {7, 13, 5, "small-prime"},
      {31, 33, 17, "non-multiple"},
      {64, 64, 64, "tile-aligned"},
      {100, 97, 63, "non-multiple-2"},
      {128, 256, 512, "rect-large-K"},
      {512, 512, 512, "large"},
      {1023, 513, 257, "large-odd"},
      // v5 tile-stress shapes (BM=BN=128, BK=8): non-multiples of 128/8,
      // tiny/prime K, K=1 edge, just-over-128 edges (bounds + zero-pad paths).
      {5, 3, 7, "k-537"},
      {9, 9, 9, "k-9"},
      {16, 16, 1, "k-one"},
      {65, 129, 9, "nonmult-128"},
      {130, 130, 130, "just-over-128"},
      {33, 17, 129, "k-129"},
      // v6 vector-path stress: full 128-tiles (fast path) and mixed
      // full/edge tiles (fast + fallback in one launch).
      {128, 128, 128, "tile128-full"},
      {200, 200, 200, "mixed-vec"},
  };
  // Magnitude coverage: second pass reuses a subset with extreme values.
  const Shape mag_shapes[] = {{32, 32, 64, "large-val"}, {32, 32, 64, "small-val"}};

  using Launcher = void (*)(const float*, const float*, float*, int, int, int);
  const struct {
    const char* name;
    Launcher fn;
  } kernels[] = {{"v0_naive", kern::launch_v0_naive},
                 {"v1_coalesced", kern::launch_v1_coalesced},
                 {"v2_tiled", kern::launch_v2_tiled},
                 {"v3_tiled_vectorized", kern::launch_v3_tiled_vectorized},
                 {"v4_coarsened", kern::launch_v4_coarsened},
                 {"v5_regtile", kern::launch_v5_regtile},
                 {"v6_vec_transposed", kern::launch_v6_vec_transposed}};

  int total_bad = 0;
#ifdef KERN_HAS_CUBLAS
  cublasHandle_t cublas_h{};
  bool cublas_ok = (cublasCreate(&cublas_h) == CUBLAS_STATUS_SUCCESS);
  if (!cublas_ok)
    std::printf("cublas unavailable at runtime; baseline checks skipped\n");
  else if (cublasSetMathMode(cublas_h, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS)
    std::printf("cublas math-mode pin failed; baseline values use default math\n");
#endif
  std::mt19937 rng(static_cast<unsigned>(seed));
  for (const auto& s : shapes) {
    std::printf("shape M=%d N=%d K=%d (%s):\n", s.M, s.N, s.K, s.tag);
    std::vector<float> hA(static_cast<std::size_t>(s.M) * s.K),
        hB(static_cast<std::size_t>(s.K) * s.N);
    fill(rng, hA, -1.0f, 1.0f);
    fill(rng, hB, -1.0f, 1.0f);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, static_cast<std::size_t>(s.M) * s.N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (const auto& k : kernels) {
      total_bad += check_one(k.name, k.fn, dA, dB, dC, hA, hB, s.M, s.N, s.K, seed);
    }
#ifdef KERN_HAS_CUBLAS
    if (cublas_ok)
      total_bad += check_cublas(cublas_h, dA, dB, dC, hA, hB, s.M, s.N, s.K, seed);
#endif
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }
  // Extreme magnitudes: very large (1e10, tests rel tolerance) and very small
  // (1e-10, tests abs tolerance floor). Justification: float has ~7 decimal
  // digits; rel check must dominate at 1e10, abs check at 1e-10.
  {
    std::vector<std::pair<const char*, std::pair<float, float>>> mags = {
        {"large-val", {1e9f, 1e10f}}, {"small-val", {1e-11f, 1e-10f}}};
    for (std::size_t mi = 0; mi < mags.size(); ++mi) {
      const auto& s = mag_shapes[mi];
      std::printf("shape M=%d N=%d K=%d (%s, range [%g,%g]):\n", s.M, s.N, s.K, s.tag,
                  mags[mi].second.first, mags[mi].second.second);
      std::vector<float> hA(static_cast<std::size_t>(s.M) * s.K),
          hB(static_cast<std::size_t>(s.K) * s.N);
      fill(rng, hA, mags[mi].second.first, mags[mi].second.second);
      fill(rng, hB, mags[mi].second.first, mags[mi].second.second);
      float *dA = nullptr, *dB = nullptr, *dC = nullptr;
      CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dC, static_cast<std::size_t>(s.M) * s.N * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
      for (const auto& k : kernels) {
        total_bad += check_one(k.name, k.fn, dA, dB, dC, hA, hB, s.M, s.N, s.K, seed);
      }
#ifdef KERN_HAS_CUBLAS
      if (cublas_ok)
        total_bad += check_cublas(cublas_h, dA, dB, dC, hA, hB, s.M, s.N, s.K, seed);
#endif
      CUDA_CHECK(cudaFree(dA));
      CUDA_CHECK(cudaFree(dB));
      CUDA_CHECK(cudaFree(dC));
    }
  }

  std::printf("[kern_correctness] %s (bad=%d)\n", total_bad == 0 ? "ALL PASS" : "FAILURES",
              total_bad);
#ifdef KERN_HAS_CUBLAS
  if (cublas_ok)
    cublasDestroy(cublas_h);
#endif
  if (!out.empty()) {
    // Minimal JSON log for the record (correctness evidence, not timing).
    FILE* f = std::fopen(out.c_str(), "w");
    if (f) {
      std::fprintf(f, "{\"seed\":%d,\"bad\":%d,\"result\":\"%s\"}\n", seed, total_bad,
                   total_bad == 0 ? "pass" : "fail");
      std::fclose(f);
    }
  }
  return total_bad == 0 ? 0 : 1;
}
