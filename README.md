# kern — SGEMM optimization ladder (CUDA)

What: single-precision matrix multiply `C = A·B` (row-major) optimized step by step —
`v0` naive → `v1` coalesced → `v2` shared-memory tiled → `v3` tiled + vectorized →
`v4` thread-coarsened (2×2 outputs/thread) → `v5` register-blocked (128-tile,
8×8/thread) — plus a cuBLAS vendor baseline.
Why: a small, reproducible study of GPU performance work where every
claim traces to a committed log file.

Status: **measured on RTX 5070 (sm_120), 2026-10-09.** Correctness gate ALL PASS
(126 checks: 18 shapes × (6 kernels + cuBLAS)), Compute Sanitizer
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
- Headline result (1024³, medians, reported run K; runs J + F/G/H/I disclosed
  in `docs/WRITEUP.md` §12):
  `v0` 2.49ms → `v1` 2.28ms → `v2` 1.13ms → `v3` 1.31ms → `v4` 0.64ms →
  **`v5` 0.43ms** (4997 GFLOPS, 38% of the 13.13 TFLOPS estimated FP32 peak) →
  cuBLAS 0.219ms (9804 GFLOPS, 75% of est. peak). Details + caveats:
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
| 1024x1024x1024 | v0_naive | 2.4937 | 861.2 | 5.0 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v1_coalesced | 2.2847 | 939.9 | 5.5 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v2_tiled | 1.1306 | 1899.5 | 11.1 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v3_tiled_vectorized | 1.3086 | 1641.0 | 9.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v4_coarsened | 0.6357 | 3378.2 | 19.8 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v5_regtile | 0.4298 | 4996.9 | 29.3 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | cublas | 0.2190 | 9804.1 | 57.4 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x512 | v0_naive | 1.6059 | 668.6 | 5.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v1_coalesced | 1.1434 | 939.1 | 7.3 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v2_tiled | 0.5964 | 1800.5 | 14.1 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v3_tiled_vectorized | 0.5954 | 1803.4 | 14.1 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v4_coarsened | 0.2673 | 4017.5 | 31.4 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v5_regtile | 0.2021 | 5312.6 | 41.5 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | cublas | 0.1120 | 9587.0 | 74.9 | `bench_M1024_N1024_K512.json` |
| 256x256x256 | v0_naive | 0.0696 | 482.3 | 11.3 | `bench_M256_N256_K256.json` |
| 256x256x256 | v1_coalesced | 0.0350 | 959.4 | 22.5 | `bench_M256_N256_K256.json` |
| 256x256x256 | v2_tiled | 0.0282 | 1188.9 | 27.9 | `bench_M256_N256_K256.json` |
| 256x256x256 | v3_tiled_vectorized | 0.0285 | 1175.5 | 27.6 | `bench_M256_N256_K256.json` |
| 256x256x256 | v4_coarsened | 0.0181 | 1852.6 | 43.4 | `bench_M256_N256_K256.json` |
| 256x256x256 | v5_regtile | 0.0554 | 605.4 | 14.2 | `bench_M256_N256_K256.json` |
| 256x256x256 | cublas | 0.0203 | 1656.5 | 38.8 | `bench_M256_N256_K256.json` |
| 512x1024x1024 | v0_naive | 1.2746 | 842.4 | 6.6 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v1_coalesced | 1.1522 | 931.9 | 7.3 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v2_tiled | 0.6127 | 1752.6 | 13.7 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v3_tiled_vectorized | 0.6133 | 1750.8 | 13.7 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v4_coarsened | 0.2810 | 3820.8 | 29.9 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v5_regtile | 0.1874 | 5730.9 | 44.8 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | cublas | 0.1100 | 9765.5 | 76.3 | `bench_M512_N1024_K1024.json` |
| 512x512x512 | v0_naive | 0.3375 | 795.3 | 9.3 | `bench_M512_N512_K512.json` |
| 512x512x512 | v1_coalesced | 0.2582 | 1039.7 | 12.2 | `bench_M512_N512_K512.json` |
| 512x512x512 | v2_tiled | 0.1782 | 1506.0 | 17.6 | `bench_M512_N512_K512.json` |
| 512x512x512 | v3_tiled_vectorized | 0.1770 | 1516.7 | 17.8 | `bench_M512_N512_K512.json` |
| 512x512x512 | v4_coarsened | 0.0818 | 3283.2 | 38.5 | `bench_M512_N512_K512.json` |
| 512x512x512 | v5_regtile | 0.0941 | 2852.3 | 33.4 | `bench_M512_N512_K512.json` |
| 512x512x512 | cublas | 0.0389 | 6904.2 | 80.9 | `bench_M512_N512_K512.json` |
<!-- RESULTS-END -->

Plots: `results/plot_gflops.png` (median GFLOPS vs shape per kernel, generated by
`python3 scripts/plot_results.py` from the bench JSONs — re-run the script after
any new bench run and commit the figure).

## Key lessons (measured — see `docs/WRITEUP.md` for evidence)

- Occupancy + work-per-thread beat wider loads: `v4` (256-thread blocks, 2×2
  outputs/thread) is 1.8× faster than `v2` at 1024³ (0.64 vs 1.13ms); `v3`'s
  float4 loads never beat `v2` beyond noise (v3 1.31 vs v2 1.13ms at 1024³) —
  a recorded negative result.
- Bigger tiles win big problems, starve small ones: `v5` (128-tile, 8×8/thread)
  is 1.5× faster than `v4` at 1024³ (0.43 vs 0.64ms) but loses at 512³ and below
  (0.33× at 256³ — 4 blocks can't fill the GPU).
- The vendor baseline earned its place twice: it caught our wrong-math OP_T
  emulation (fixed to OP_N, validated value-by-value), and it leads the best
  custom kernel 2.0× at 1024³ (9804 vs 4997 GFLOPS).
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
