# kern — SGEMM optimization ladder (CUDA)

What: single-precision matrix multiply `C = A·B` (row-major) optimized step by step —
`v0` naive → `v1` coalesced → `v2` shared-memory tiled → `v3` tiled + vectorized →
`v4` thread-coarsened (2×2 outputs/thread) — plus a cuBLAS vendor baseline.
Why: a small, reproducible study of GPU performance work where every
claim traces to a committed log file.

Status: **measured on RTX 5070 (sm_120), 2026-10-09.** Correctness gate ALL PASS
(72 checks: 12 shapes × (5 kernels + cuBLAS)), Compute Sanitizer
clean (memcheck/racecheck/synccheck/initcheck), bench run twice
with drift check + adjudicating third run. Profiler counters are blocked on this
WDDM machine (`ERR_NVGPUCTRPERM`, evidence in `results/ncu/`), so performance
claims rest on `cudaEvent` timing plus labeled analytical reasoning.

## Hardware used

- Bench + scaffolding machine: NVIDIA GeForce RTX 5070 (sm_120, Blackwell),
  driver 617.14, AMD Ryzen AI 7 350, Windows 64-bit, WDDM (display-attached —
  expect scheduling noise; hence medians, repeats, drift checks). CUDA 13.0 +
  MSVC 19.44, CMake 4.4.3, `ncu`/`compute-sanitizer` 2025.3.1.0, no `nsys`.
  Full record: `docs/ENVIRONMENT.md`.
- Headline result (1024³, medians, run E of D/E + adjudicating run C):
  `v0` 2.48ms → `v1` 2.26ms → `v2` 1.17ms → `v3` 1.22ms → **`v4` 0.62ms**
  (3457 GFLOPS, 26% of the 13.13 TFLOPS estimated FP32 peak) → cuBLAS 0.226ms
  (9508 GFLOPS, 72% of est. peak). Details + caveats: `docs/WRITEUP.md`.

## Build, run, reproduce (one command each)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKERN_CUDA_ARCH=sm_120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

(`sm_120` matches the RTX 5070 inference; override per GPU, e.g. `sm_80` for A100.
Needs CUDA 12.8 or newer for 50-series (verified here with CUDA 13.0 + MSVC 19.44).
CPU-only build works with no toolkit.
On Windows with VS 2022 Build Tools, configure inside a developer prompt
(`VsDevCmd -arch=amd64`) with `-G "NMake Makefiles"` and an explicit
`-DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe"`.
With MSYS2/MinGW and no Visual Studio, use `-G "MinGW Makefiles"` instead;
the default `NMake Makefiles` generator fails without `nmake`.)

```bash
# Correctness gate (all kernels vs double-accumulated CPU reference, fixed seeds)
./build/kern_correctness --seed 1234 --out results/correctness_seed1234.json

# Sanitizer (clean logs required; see scripts/run_sanitizer.sh|.bat)
bash scripts/run_sanitizer.sh ./build/kern_correctness

# Benchmarks — run EACH at least twice; investigate before reporting if the two
# medians differ by more than five percent
./build/kern_bench --all --seed 0
./build/kern_bench --all --seed 0
python3 scripts/generate_tables.py --write
python3 scripts/claims_audit.py
```

CPU-only checks (run anywhere, including CI):

```bash
python3 tests/test_correctness_cpu.py
python3 scripts/claims_audit.py
python3 scripts/generate_tables.py --check
```

## Results (generated from `results/`, never typed by hand)

<!-- RESULTS-BEGIN -->
| Shape (MxNxK) | Kernel | Median (ms) | GFLOPS | GB/s | Source |
|---|---|---|---|---|---|
| 1024x1024x1024 | v0_naive | 2.4838 | 864.6 | 5.1 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v1_coalesced | 2.2586 | 950.8 | 5.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v2_tiled | 1.1672 | 1839.8 | 10.8 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v3_tiled_vectorized | 1.2242 | 1754.3 | 10.3 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v4_coarsened | 0.6212 | 3456.7 | 20.3 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | cublas | 0.2259 | 9508.2 | 55.7 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x512 | v0_naive | 1.2480 | 860.3 | 6.7 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v1_coalesced | 1.1280 | 951.9 | 7.4 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v2_tiled | 0.6012 | 1785.9 | 14.0 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v3_tiled_vectorized | 0.5912 | 1816.3 | 14.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v4_coarsened | 0.2796 | 3840.1 | 30.0 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | cublas | 0.1128 | 9519.0 | 74.4 | `bench_M1024_N1024_K512.json` |
| 256x256x256 | v0_naive | 0.0728 | 461.1 | 10.8 | `bench_M256_N256_K256.json` |
| 256x256x256 | v1_coalesced | 0.0368 | 912.6 | 21.4 | `bench_M256_N256_K256.json` |
| 256x256x256 | v2_tiled | 0.0279 | 1203.9 | 28.2 | `bench_M256_N256_K256.json` |
| 256x256x256 | v3_tiled_vectorized | 0.0293 | 1144.7 | 26.8 | `bench_M256_N256_K256.json` |
| 256x256x256 | v4_coarsened | 0.0180 | 1865.8 | 43.7 | `bench_M256_N256_K256.json` |
| 256x256x256 | cublas | 0.0200 | 1680.4 | 39.4 | `bench_M256_N256_K256.json` |
| 512x1024x1024 | v0_naive | 1.2771 | 840.8 | 6.6 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v1_coalesced | 1.1253 | 954.1 | 7.5 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v2_tiled | 0.6288 | 1707.7 | 13.3 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v3_tiled_vectorized | 0.6177 | 1738.2 | 13.6 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v4_coarsened | 0.2762 | 3888.1 | 30.4 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | cublas | 0.1165 | 9215.7 | 72.0 | `bench_M512_N1024_K1024.json` |
| 512x512x512 | v0_naive | 0.3369 | 796.7 | 9.3 | `bench_M512_N512_K512.json` |
| 512x512x512 | v1_coalesced | 0.2583 | 1039.2 | 12.2 | `bench_M512_N512_K512.json` |
| 512x512x512 | v2_tiled | 0.1791 | 1498.5 | 17.6 | `bench_M512_N512_K512.json` |
| 512x512x512 | v3_tiled_vectorized | 0.1774 | 1513.1 | 17.7 | `bench_M512_N512_K512.json` |
| 512x512x512 | v4_coarsened | 0.0815 | 3292.2 | 38.6 | `bench_M512_N512_K512.json` |
| 512x512x512 | cublas | 0.0392 | 6853.4 | 80.3 | `bench_M512_N512_K512.json` |
<!-- RESULTS-END -->

Plots: `results/plot_gflops.png` (median GFLOPS vs shape per kernel, generated by
`python3 scripts/plot_results.py` from the bench JSONs — re-run the script after
any new bench run and commit the figure).

## Key lessons (measured — see `docs/WRITEUP.md` for evidence)

- Occupancy + work-per-thread beat wider loads: `v4` (256-thread blocks, 2×2
  outputs/thread) is 1.9× faster than `v2` at 1024³ (0.62 vs 1.17ms); `v3`'s
  float4 loads are within ±5% of `v2` everywhere — a recorded negative result.
- The vendor baseline earned its place twice: it caught our wrong-math OP_T
  emulation (fixed to OP_N, validated value-by-value), and it leads the best
  custom kernel 2.7× at 1024³ (9508 vs 3457 GFLOPS).
- Protocol before speed: `cudaEvent` kernel-only timing, warmup plus timed iters,
  median/min/p95, identical inputs, JSON evidence with environment metadata,
  repeat runs with a scripted >5% drift check (`scripts/check_repeat.py`).
- Correctness-first ladder: each version is its own file with bounds-checked
  arbitrary sizes, so `git bisect` / per-version benching stays reproducible.

## Limitations

- GitHub Actions cannot run GPU code (no GPU runners): CI runs formatting, Python
  tests, and CPU-side checks only.
- Update 2026-10-09: GPU execution works on this machine (driver 617.14;
  `kern_correctness` ALL PASS). The earlier Smart App Control block no longer
  stops execution; the prior hang was the v3 divergent-barrier bug (fixed).
  See `docs/ENVIRONMENT.md` and `LOG.md`.
- WDDM blocks `ncu` counters on this machine (`ERR_NVGPUCTRPERM`; attempt log in
  `results/ncu/` — enabling counters needs admin action, out of scope), so
  profiler evidence stays `NOT RUN` and claims rest on timing plus labeled
  analytical reasoning.
- `v2`/`v3` use 32x32 thread blocks (1024 threads, the architectural max):
  portable but occupancy-limited — which is exactly what `v4` (16x16 threads ×
  2x2 outputs) tested and confirmed (see `docs/WRITEUP.md`).

## References

See `docs/REFERENCES.md` (programming guide, best-practices guide, matrixMul/tiled
pattern, cuBLAS, Nsight Compute, Blackwell toolkit notes). Adapted ideas are also
cited in code comments.
