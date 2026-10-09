# kern — SGEMM optimization ladder (CUDA)

What: single-precision matrix multiply `C = A·B` (row-major) optimized step by step —
`v0` naive → `v1` coalesced → `v2` shared-memory tiled → `v3` tiled + vectorized →
`v4` thread-coarsened (2×2 outputs/thread) → `v5` register-blocked (128-tile,
8×8/thread) → `v6` vectorized + transposed — plus a cuBLAS vendor baseline.
Why: a small, reproducible study of GPU performance work where every
claim traces to a committed log file.

Status: **measured on RTX 5070 (sm_120), 2026-10-09.** Correctness gate ALL PASS
(160 checks: 20 shapes × (7 kernels + cuBLAS)), Compute Sanitizer
clean (memcheck/racecheck/synccheck/initcheck), bench run twice
with drift check + adjudicating runs. Profiler counters are blocked on this
WDDM machine (`ERR_NVGPUCTRPERM`, evidence in `results/ncu/`), so performance
claims rest on `cudaEvent` timing plus labeled analytical reasoning.

## Hardware used

- Bench + scaffolding machine: NVIDIA GeForce RTX 5070 (sm_120, Blackwell),
  driver 617.14, AMD Ryzen AI 7 350, Windows 64-bit, WDDM (display-attached —
  expect scheduling noise; hence medians, repeats, drift checks). CUDA 13.0 +
  MSVC 19.44, CMake 4.4.3, `ncu`/`compute-sanitizer` 2025.3.1.0, no `nsys`.
  Full record: `docs/ENVIRONMENT.md`.
- Headline result (1024³, medians, reported run M; runs L + J/K/F/G/H/I disclosed
  in `docs/WRITEUP.md` §13):
  `v0` 2.48ms → `v1` 2.26ms → `v2` 1.13ms → `v3` 1.26ms → `v4` 0.65ms →
  `v5` 0.41ms → **`v6` 0.305ms** (7034 GFLOPS, 54% of the 13.13 TFLOPS estimated FP32 peak) →
  cuBLAS 0.221ms (9699 GFLOPS, 74% of est. peak). Details + caveats:
  `docs/WRITEUP.md`.

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
| 1024x1024x1024 | v0_naive | 2.4800 | 865.9 | 5.1 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v1_coalesced | 2.2563 | 951.8 | 5.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v2_tiled | 1.1277 | 1904.2 | 11.2 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v3_tiled_vectorized | 1.2580 | 1707.1 | 10.0 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v4_coarsened | 0.6544 | 3281.4 | 19.2 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v5_regtile | 0.4096 | 5242.5 | 30.7 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v6_vec_transposed | 0.3053 | 7033.7 | 41.2 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | cublas | 0.2214 | 9699.2 | 56.8 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x512 | v0_naive | 1.6069 | 668.2 | 5.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v1_coalesced | 1.1383 | 943.3 | 7.4 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v2_tiled | 0.5931 | 1810.5 | 14.1 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v3_tiled_vectorized | 0.5909 | 1817.2 | 14.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v4_coarsened | 0.2668 | 4024.3 | 31.4 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v5_regtile | 0.2012 | 5337.1 | 41.7 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v6_vec_transposed | 0.1608 | 6678.8 | 52.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | cublas | 0.1083 | 9918.5 | 77.5 | `bench_M1024_N1024_K512.json` |
| 256x256x256 | v0_naive | 0.0724 | 463.2 | 10.9 | `bench_M256_N256_K256.json` |
| 256x256x256 | v1_coalesced | 0.0356 | 943.0 | 22.1 | `bench_M256_N256_K256.json` |
| 256x256x256 | v2_tiled | 0.0277 | 1212.2 | 28.4 | `bench_M256_N256_K256.json` |
| 256x256x256 | v3_tiled_vectorized | 0.0287 | 1170.3 | 27.4 | `bench_M256_N256_K256.json` |
| 256x256x256 | v4_coarsened | 0.0180 | 1865.8 | 43.7 | `bench_M256_N256_K256.json` |
| 256x256x256 | v5_regtile | 0.0544 | 617.2 | 14.5 | `bench_M256_N256_K256.json` |
| 256x256x256 | v6_vec_transposed | 0.0469 | 714.8 | 16.8 | `bench_M256_N256_K256.json` |
| 256x256x256 | cublas | 0.0205 | 1635.8 | 38.3 | `bench_M256_N256_K256.json` |
| 512x1024x1024 | v0_naive | 1.2743 | 842.6 | 6.6 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v1_coalesced | 1.1385 | 943.1 | 7.4 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v2_tiled | 0.6090 | 1763.1 | 13.8 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v3_tiled_vectorized | 0.6142 | 1748.3 | 13.7 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v4_coarsened | 0.2782 | 3859.5 | 30.2 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v5_regtile | 0.1862 | 5765.4 | 45.0 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v6_vec_transposed | 0.1517 | 7077.5 | 55.3 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | cublas | 0.1099 | 9771.2 | 76.3 | `bench_M512_N1024_K1024.json` |
| 512x512x512 | v0_naive | 0.3376 | 795.2 | 9.3 | `bench_M512_N512_K512.json` |
| 512x512x512 | v1_coalesced | 0.2601 | 1032.2 | 12.1 | `bench_M512_N512_K512.json` |
| 512x512x512 | v2_tiled | 0.1852 | 1449.8 | 17.0 | `bench_M512_N512_K512.json` |
| 512x512x512 | v3_tiled_vectorized | 0.1776 | 1511.7 | 17.7 | `bench_M512_N512_K512.json` |
| 512x512x512 | v4_coarsened | 0.0826 | 3248.9 | 38.1 | `bench_M512_N512_K512.json` |
| 512x512x512 | v5_regtile | 0.0947 | 2834.9 | 33.2 | `bench_M512_N512_K512.json` |
| 512x512x512 | v6_vec_transposed | 0.0791 | 3392.1 | 39.8 | `bench_M512_N512_K512.json` |
| 512x512x512 | cublas | 0.0397 | 6754.1 | 79.2 | `bench_M512_N512_K512.json` |
<!-- RESULTS-END -->

Plots: `results/plot_gflops.png` (median GFLOPS vs shape per kernel, generated by
`python3 scripts/plot_results.py` from the bench JSONs — re-run the script after
any new bench run and commit the figure).

## Key lessons (measured — see `docs/WRITEUP.md` for evidence)

- Occupancy + work-per-thread beat wider loads: `v4` (256-thread blocks, 2×2
  outputs/thread) is 1.8× faster than `v2` at 1024³ (0.65 vs 1.13ms); `v3`'s
  float4 loads never beat `v2` beyond noise (v3 1.26 vs v2 1.13ms at 1024³) —
  a recorded negative result.
- Bigger tiles win big problems, starve small ones: `v5` (128-tile, 8×8/thread)
  is 1.6× faster than `v4` at 1024³ (0.41 vs 0.65ms) but loses at 512³ and below
  (0.33× at 256³ — 4 blocks can't fill the GPU).
- Layout + vectors stack on top: `v6` (float4 fills, transposed A tile) is 1.3×
  faster than `v5` on every shape including 256³ (0.305 vs 0.41ms at 1024³) —
  shared-memory layout was second-order, but not zero.
- The vendor baseline earned its place twice: it caught our wrong-math OP_T
  emulation (fixed to OP_N, validated value-by-value), and it leads the best
  custom kernel 1.4× at 1024³ (9699 vs 7034 GFLOPS).
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
