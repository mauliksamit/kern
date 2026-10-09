// Benchmark protocol.
// - cudaEvent around the kernel ONLY (no H2D/D2H inside the timed region).
// - >=10 warmup, >=100 timed iterations (fewer if one run exceeds 100 ms, noted in JSON).
// - Reports median, min, p95 over identical inputs across versions.
// - Derived metrics (defined here):
//     GFLOPS = 2*M*N*K / time_s / 1e9   (2 flops per multiply-add)
//     GB/s   = (M*K + K*N + M*N) * 4B / time_s / 1e9  (ideal DRAM traffic, no reuse)
// - Theoretical peak (FP32 + bandwidth) queried from cudaGetDeviceProperties and
//   recorded in JSON; achieved % computed against it.
// - Re-run each benchmark at least twice; if medians differ by >5%, flag it
//   (compare two result files with scripts/check_repeat.py logic / eyeball).
// - Raw per-iteration timings saved as JSON in results/ with metadata:
//   git commit, env (GPU name, CC, driver), compile flags, sizes, seed.
// - Vendor baseline: cuBLAS sgemm timed identically (where CUDAToolkit found).
// Usage:
//   kern_bench --M 1024 --N 1024 --K 1024 --seed 0 --out results/bench_M1024_N1024_K1024.json
//   kern_bench --all  (runs 256/512/1024 square + 2 rect shapes, one JSON each)
#include "kern/cuda_check.h"
#include "kern/kernels.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// CUDA 13 removed clock fields (clockRate, memoryClockRate) from cudaDeviceProp.
// Clocks are queried via the driver API (cuDeviceGetAttribute) instead; the
// raw kHz values are stored in the JSON so the peak computation stays auditable.
#include <cuda.h>

#ifdef KERN_HAS_CUBLAS
#include <cublas_v2.h>
#define KERN_BENCH_HAS_CUBLAS 1
#else
#define KERN_BENCH_HAS_CUBLAS 0
#endif

namespace {

struct Timing {
  std::string name;
  std::vector<float> iters_ms; // per-iteration kernel time.
  double median_ms = 0, min_ms = 0, p95_ms = 0, mean_ms = 0;
  double gflops = 0, gbps = 0; // derived from median.
  bool failed = false;         // true if the backend reported failure (no valid timing).
};

double
percentile(std::vector<float> v, double p) {
  std::sort(v.begin(), v.end());
  if (v.empty())
    return 0;
  std::size_t i = static_cast<std::size_t>(p * (v.size() - 1));
  return v[i];
}

std::string
git_commit() {
  // Best-effort: baked at configure time would be nicer; runtime popen keeps
  // the binary portable. "unknown" if git is absent (e.g. exported tarball).
  // Overridable with --commit (Phase 0 evidence rule: never report "unknown"
  // when the revision is known).
  FILE* p = nullptr;
#ifdef _WIN32
  p = _popen("git rev-parse --short HEAD 2>NUL", "r");
#else
  p = popen("git rev-parse --short HEAD 2>/dev/null", "r");
#endif
  char buf[64] = {0};
  if (p) {
    fgets(buf, sizeof(buf), p);
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
  }
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
    s.pop_back();
  return s.empty() ? "unknown" : s;
}

// Read-only nvidia-smi query (observe only: no clocks, power, or device-state
// changes). Returns trimmed stdout or "unknown".
std::string
smi_query(const char* field) {
  std::string cmd =
      std::string("nvidia-smi --query-gpu=") + field + " --format=csv,noheader,nounits";
  FILE* p = nullptr;
#ifdef _WIN32
  p = _popen(cmd.c_str(), "r");
#else
  p = popen(cmd.c_str(), "r");
#endif
  char buf[256] = {0};
  if (p) {
    fgets(buf, sizeof(buf), p);
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
  }
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
    s.pop_back();
  while (!s.empty() && s.front() == ' ')
    s.erase(s.begin());
  return s.empty() ? "unknown" : s;
}

struct EnvSample {
  std::string sm_mhz = "unknown";     // clocks.sm (reported, not set)
  std::string max_sm_mhz = "unknown"; // clocks.max.sm (reported max, not set)
  std::string power_w = "unknown";    // power.draw
  std::string temp_c = "unknown";     // temperature.gpu
  std::string power_limit_w = "unknown";
};

EnvSample
sample_env() {
  EnvSample e;
  e.sm_mhz = smi_query("clocks.sm");
  e.max_sm_mhz = smi_query("clocks.max.sm");
  e.power_w = smi_query("power.draw");
  e.temp_c = smi_query("temperature.gpu");
  e.power_limit_w = smi_query("power.limit");
  return e;
}

// Exact compile flags, baked by CMake (KERN_COMPILE_FLAGS define) so the JSON
// records what actually built the binary. Falls back to "unknown", never guessed.
const char*
compile_flags_str() {
#ifdef KERN_COMPILE_FLAGS
#define KERN_STR_(x) #x
#define KERN_STR(x) KERN_STR_(x)
  return KERN_STR(KERN_COMPILE_FLAGS);
#else
  return "unknown (no KERN_COMPILE_FLAGS define)";
#endif
}

void
time_kernel(const char* name, void (*launch)(const float*, const float*, float*, int, int, int),
            const float* dA, const float* dB, float* dC, int M, int N, int K, Timing& out) {
  out.name = name;
  // Warmup (>=10).
  for (int i = 0; i < 10; ++i) {
    launch(dA, dB, dC, M, N, K);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Probe one iteration to decide count (fewer if one run > 100 ms).
  CUDA_CHECK(cudaEventRecord(start));
  launch(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float probe = 0;
  CUDA_CHECK(cudaEventElapsedTime(&probe, start, stop));
  int iters = (probe > 100.0f) ? 10 : 100;
  bool reduced = (iters < 100);

  out.iters_ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    out.iters_ms.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<float> s = out.iters_ms;
  std::sort(s.begin(), s.end());
  out.min_ms = s.front();
  out.median_ms = percentile(s, 0.5);
  out.p95_ms = percentile(s, 0.95);
  out.mean_ms = std::accumulate(s.begin(), s.end(), 0.0) / static_cast<double>(s.size());
  double t_s = out.median_ms / 1e3;
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double bytes = ((double)M * K + (double)K * N + (double)M * N) * 4.0;
  out.gflops = flops / t_s / 1e9;
  out.gbps = bytes / t_s / 1e9;
  std::printf("  %-18s median=%9.4f ms  min=%9.4f  p95=%9.4f  GFLOPS=%9.1f  GB/s=%8.1f%s\n", name,
              out.median_ms, out.min_ms, out.p95_ms, out.gflops, out.gbps,
              reduced ? "  [iters=10: single run >100ms]" : "");
}

#if KERN_BENCH_HAS_CUBLAS
// Returns true on success; on cuBLAS failure returns false with err set (the
// caller records a "failed" JSON entry instead of aborting, so one bad shape
// cannot destroy the other shapes' timings in --all mode).
bool
time_cublas(cublasHandle_t h, const float* dA, const float* dB, float* dC, int M, int N, int K,
            Timing& out, std::string& err) {
  // cuBLAS is column-major. Row-major C = A*B (A: MxK, B: KxN) is equivalent to
  // col-major C^T = B^T * A^T. A row-major MxK buffer viewed column-major IS the
  // KxM transpose, so pass the buffers swapped with CUBLAS_OP_N (no transpose
  // performed -- the "transpose" is just the change of interpretation):
  //   op(A_in=dB) is N x K = B^T (lda=N rows, need lda >= m=N: always exact),
  //   op(B_in=dA) is K x M = A^T (ldb=K rows, need ldb >= k=K: always exact),
  //   C_out=dC is N x M = C^T (ldc=N rows, need ldc >= m=N: always exact).
  // Valid for ALL shapes with no copies (fixed 2026-10-09: the original used
  // OP_T+swapped buffers, whose transposes cancel to B*A -- wrong math, caught
  // by the new cuBLAS correctness check; see LOG.md).
  out.name = "cublas";
  out.failed = false;
  const float alpha = 1.0f, beta = 0.0f;
  auto sgemm = [&]() {
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  };
  for (int i = 0; i < 10; ++i) {
    cublasStatus_t st = sgemm();
    if (st != CUBLAS_STATUS_SUCCESS) {
      err = "cublasSgemm warmup failed";
      std::fprintf(stderr, "cublasSgemm warmup failed (M=%d N=%d K=%d status=%d)\n", M, N, K,
                   (int)st);
      out.failed = true;
      out.iters_ms.clear();
      return false;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  if (sgemm() != CUBLAS_STATUS_SUCCESS) {
    err = "cublasSgemm probe failed";
    out.failed = true;
    out.iters_ms.clear();
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return false;
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float probe = 0;
  CUDA_CHECK(cudaEventElapsedTime(&probe, start, stop));
  int iters = (probe > 100.0f) ? 10 : 100;
  out.iters_ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    sgemm();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    out.iters_ms.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::vector<float> s = out.iters_ms;
  std::sort(s.begin(), s.end());
  out.min_ms = s.front();
  out.median_ms = percentile(s, 0.5);
  out.p95_ms = percentile(s, 0.95);
  out.mean_ms = std::accumulate(s.begin(), s.end(), 0.0) / (double)s.size();
  double t_s = out.median_ms / 1e3;
  out.gflops = (2.0 * M * N * K) / t_s / 1e9;
  out.gbps = ((double(M * K + K * N + M * N)) * 4.0) / t_s / 1e9;
  std::printf("  %-18s median=%9.4f ms  min=%9.4f  p95=%9.4f  GFLOPS=%9.1f  GB/s=%8.1f\n", "cublas",
              out.median_ms, out.min_ms, out.p95_ms, out.gflops, out.gbps);
  return true;
}
#endif

void
run_one(int M, int N, int K, int seed, const std::string& out_path,
        const std::string& commit_override) {
  std::printf("[bench] M=%d N=%d K=%d seed=%d\n", M, N, K, seed);
  EnvSample env_before = sample_env();
  std::mt19937 rng((unsigned)seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> hA((std::size_t)M * K), hB((std::size_t)K * N);
  for (auto& x : hA)
    x = d(rng);
  for (auto& x : hB)
    x = d(rng);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (std::size_t)M * N * sizeof(float)));
  // Copies OUTSIDE the timed region.
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<Timing> ts(8);
  time_kernel("v0_naive", kern::launch_v0_naive, dA, dB, dC, M, N, K, ts[0]);
  time_kernel("v1_coalesced", kern::launch_v1_coalesced, dA, dB, dC, M, N, K, ts[1]);
  time_kernel("v2_tiled", kern::launch_v2_tiled, dA, dB, dC, M, N, K, ts[2]);
  time_kernel("v3_tiled_vectorized", kern::launch_v3_tiled_vectorized, dA, dB, dC, M, N, K, ts[3]);
  time_kernel("v4_coarsened", kern::launch_v4_coarsened, dA, dB, dC, M, N, K, ts[4]);
  time_kernel("v5_regtile", kern::launch_v5_regtile, dA, dB, dC, M, N, K, ts[5]);
  time_kernel("v6_vec_transposed", kern::launch_v6_vec_transposed, dA, dB, dC, M, N, K, ts[6]);
#if KERN_BENCH_HAS_CUBLAS
  cublasHandle_t h;
  std::string math_mode = "cublas_unavailable";
  if (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) {
    // Fairness (Phase 0): pin FP32 math explicitly. PEDANTIC disallows any
    // reduced-precision path (including TF32 tensor math) for sgemm, so the
    // baseline cannot silently use TF32. Verified below via cublasGetMathMode.
    std::string cublas_err;
    if (cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
      math_mode = "SET_FAILED(continuing, mode unverified)";
    } else {
      cublasMath_t got = CUBLAS_PEDANTIC_MATH;
      if (cublasGetMathMode(h, &got) != CUBLAS_STATUS_SUCCESS || got != CUBLAS_PEDANTIC_MATH)
        math_mode = "GET_FAILED(or mismatch)";
      else
        math_mode = "CUBLAS_PEDANTIC_MATH(TF32 disallowed)";
    }
    if (!time_cublas(h, dA, dB, dC, M, N, K, ts[7], cublas_err)) {
      std::printf("  cublas FAILED for this shape (%s); custom-kernel timings still saved\n",
                  cublas_err.c_str());
    }
    cublasDestroy(h);
  } else {
    ts[7].name = "cublas";
    ts[7].failed = true;
    std::printf("  cublas unavailable at runtime\n");
  }
#else
  ts[7].name = "cublas";
  ts[7].failed = true;
  std::printf("  cublas not compiled in (no CUDAToolkit at build time)\n");
#endif

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // Theoretical FP32 peak estimate, computed from device properties:
  //   peak = SMs * 128 FP32 cores/SM * 2 (FMA) * SM clock.
  // 128 cores/SM holds for Ampere and later (incl. Blackwell); SM clock comes
  // from CU_DEVICE_ATTRIBUTE_CLOCK_RATE (kHz) via the driver API. Recorded as
  // informational; achieved_% uses it. Cores/SM should be re-verified against
  // the official whitepaper for future architectures before citing.
  // NOTE: no memory-bandwidth peak is estimated here: GDDR effective-rate
  // multipliers differ per memory type, so achieved GB/s is reported without a
  // % claim (see docs/WRITEUP.md limitation).
  int smClockKHz = 0, memClockKHz = 0;
  if (cuInit(0) == CUDA_SUCCESS) {
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
      cuDeviceGetAttribute(&smClockKHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
      cuDeviceGetAttribute(&memClockKHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
    }
  }
  double clock_ghz = smClockKHz / 1e6;
  double peak_fp32_tflops = prop.multiProcessorCount * 128.0 * 2.0 * clock_ghz / 1e3;

  FILE* f = fopen(out_path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", out_path.c_str());
    std::exit(1);
  }
  EnvSample env_after = sample_env();
  // Max observed SM clock across the two read-only samples (still a reported
  // value, not a measured boost curve — see peak note below).
  auto to_mhz = [](const std::string& s) -> double {
    try {
      return std::stod(s);
    } catch (...) {
      return -1.0;
    }
  };
  double sm_max_obs = std::max(to_mhz(env_before.sm_mhz), to_mhz(env_after.sm_mhz));
  std::string commit = commit_override.empty() ? git_commit() : commit_override;
  std::fprintf(f,
               "{\n  \"M\": %d, \"N\": %d, \"K\": %d, \"seed\": %d,\n"
               "  \"git_commit\": \"%s\",\n"
               "  \"device\": \"%s\", \"cc\": \"%d.%d\", \"driver\": \"%s\",\n"
               "  \"compile_flags\": \"%s\",\n"
               "  \"cublas_math_mode\": \"%s\",\n"
               "  \"warmup\": 10, \"protocol\": \"cudaEvent kernel-only; median/min/p95\",\n"
               "  \"sm_count\": %d, \"sm_clock_khz\": %d, \"mem_clock_khz\": %d, "
               "\"mem_bus_width_bits\": %d,\n"
               "  \"peak_fp32_tflops_est\": %.2f,\n"
               "  \"peak_note\": \"from reported SM clock via driver API (not measured boost "
               "clock); 128 FP32 cores/SM assumed, re-verify per whitepaper\",\n"
               "  \"sm_clock_max_observed_khz\": %.0f,\n"
               "  \"env_before\": {\"clocks_sm_mhz\": \"%s\", \"clocks_max_sm_mhz\": \"%s\", "
               "\"power_draw_w\": \"%s\", \"temp_c\": \"%s\", \"power_limit_w\": \"%s\"},\n"
               "  \"env_after\": {\"clocks_sm_mhz\": \"%s\", \"clocks_max_sm_mhz\": \"%s\", "
               "\"power_draw_w\": \"%s\", \"temp_c\": \"%s\", \"power_limit_w\": \"%s\"},\n"
               "  \"results\": [\n",
               M, N, K, seed, commit.c_str(), prop.name, prop.major, prop.minor,
               smi_query("driver_version").c_str(), compile_flags_str(), math_mode.c_str(),
               prop.multiProcessorCount, smClockKHz, memClockKHz, prop.memoryBusWidth,
               peak_fp32_tflops, sm_max_obs * 1000.0, env_before.sm_mhz.c_str(),
               env_before.max_sm_mhz.c_str(), env_before.power_w.c_str(), env_before.temp_c.c_str(),
               env_before.power_limit_w.c_str(), env_after.sm_mhz.c_str(),
               env_after.max_sm_mhz.c_str(), env_after.power_w.c_str(), env_after.temp_c.c_str(),
               env_after.power_limit_w.c_str());
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::fprintf(f,
                 "    {\"kernel\": \"%s\", \"failed\": %s, \"median_ms\": %.6f, \"min_ms\": %.6f, "
                 "\"p95_ms\": %.6f, \"mean_ms\": %.6f, \"gflops\": %.3f, \"gbps\": %.3f, "
                 "\"iters\": [",
                 ts[i].name.c_str(), ts[i].failed ? "true" : "false", ts[i].median_ms, ts[i].min_ms,
                 ts[i].p95_ms, ts[i].mean_ms, ts[i].gflops, ts[i].gbps);
    for (std::size_t j = 0; j < ts[i].iters_ms.size(); ++j) {
      std::fprintf(f, "%s%.6f", j ? ", " : "", ts[i].iters_ms[j]);
    }
    std::fprintf(f, "]}%s\n", i + 1 < ts.size() ? "," : "");
  }
  std::fprintf(f, "  ]\n}\n");
  fclose(f);
  std::printf("[bench] wrote %s (re-run at least twice; flag >5%% median drift)\n",
              out_path.c_str());

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
}

} // namespace

int
main(int argc, char** argv) {
  int M = 1024, N = 1024, K = 1024, seed = 0;
  std::string out = "results/bench_M1024_N1024_K1024.json";
  std::string commit_override; // --commit: stamp JSON when git popen cannot (else auto)
  bool all = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--M") == 0 && i + 1 < argc)
      M = atoi(argv[++i]);
    else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc)
      N = atoi(argv[++i]);
    else if (strcmp(argv[i], "--K") == 0 && i + 1 < argc)
      K = atoi(argv[++i]);
    else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
      seed = atoi(argv[++i]);
    else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
      out = argv[++i];
    else if (strcmp(argv[i], "--commit") == 0 && i + 1 < argc)
      commit_override = argv[++i];
    else if (strcmp(argv[i], "--all") == 0)
      all = true;
  }
  if (!all) {
    run_one(M, N, K, seed, out, commit_override);
    return 0;
  }
  const int sq[] = {256, 512, 1024};
  for (int s : sq) {
    char path[256];
    snprintf(path, sizeof(path), "results/bench_M%d_N%d_K%d.json", s, s, s);
    run_one(s, s, s, seed, path, commit_override);
  }
  run_one(1024, 1024, 512, seed, "results/bench_M1024_N1024_K512.json", commit_override);
  run_one(512, 1024, 1024, seed, "results/bench_M512_N1024_K1024.json", commit_override);
  return 0;
}
