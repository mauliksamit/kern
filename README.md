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
- Headline result (1024³, medians, reported run N; runs O/P + J/K/L/M/F/G/H/I disclosed
  in `docs/WRITEUP.md` §13):
  `v0` 2.48ms → `v1` 2.27ms → `v2` 1.12ms → `v3` 1.31ms → `v4` 0.64ms →
  `v5` 0.425ms → **`v6` 0.306ms** (7026 GFLOPS, 54% of the 13.13 TFLOPS estimated FP32 peak) →
  cuBLAS 0.205ms (10456 GFLOPS, 80% of est. peak). Details + caveats:
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
| 1024x1024x1024 | v0_naive | 2.4838 | 864.6 | 5.1 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v1_coalesced | 2.2709 | 945.6 | 5.5 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v2_tiled | 1.1202 | 1917.1 | 11.2 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v3_tiled_vectorized | 1.3113 | 1637.7 | 9.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v4_coarsened | 0.6391 | 3360.3 | 19.7 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v5_regtile | 0.4251 | 5052.2 | 29.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v6_vec_transposed | 0.3057 | 7025.6 | 41.2 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | cublas | 0.2054 | 10456.4 | 61.3 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x512 | v0_naive | 1.6116 | 666.3 | 5.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v1_coalesced | 1.1558 | 929.0 | 7.3 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v2_tiled | 0.6020 | 1783.5 | 13.9 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v3_tiled_vectorized | 0.5946 | 1805.9 | 14.1 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v4_coarsened | 0.2708 | 3965.3 | 31.0 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v5_regtile | 0.2053 | 5230.6 | 40.9 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v6_vec_transposed | 0.1632 | 6580.6 | 51.4 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | cublas | 0.1101 | 9748.5 | 76.2 | `bench_M1024_N1024_K512.json` |
| 2048x2048x2048 | v0_naive | 22.6952 | 757.0 | 2.2 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v1_coalesced | 22.1483 | 775.7 | 2.3 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v2_tiled | 9.8460 | 1744.8 | 5.1 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v3_tiled_vectorized | 10.5478 | 1628.8 | 4.8 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v4_coarsened | 4.6571 | 3689.0 | 10.8 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v5_regtile | 2.7927 | 6151.8 | 18.0 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v6_vec_transposed | 2.2547 | 7619.5 | 22.3 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | cublas | 1.5441 | 11125.9 | 32.6 | `bench_M2048_N2048_K2048.json` |
| 256x256x256 | v0_naive | 0.0729 | 460.5 | 10.8 | `bench_M256_N256_K256.json` |
| 256x256x256 | v1_coalesced | 0.0348 | 965.5 | 22.6 | `bench_M256_N256_K256.json` |
| 256x256x256 | v2_tiled | 0.0275 | 1219.3 | 28.6 | `bench_M256_N256_K256.json` |
| 256x256x256 | v3_tiled_vectorized | 0.0285 | 1178.2 | 27.6 | `bench_M256_N256_K256.json` |
| 256x256x256 | v4_coarsened | 0.0177 | 1892.7 | 44.4 | `bench_M256_N256_K256.json` |
| 256x256x256 | v5_regtile | 0.0558 | 600.9 | 14.1 | `bench_M256_N256_K256.json` |
| 256x256x256 | v6_vec_transposed | 0.0470 | 713.8 | 16.7 | `bench_M256_N256_K256.json` |
| 256x256x256 | cublas | 0.0213 | 1576.8 | 37.0 | `bench_M256_N256_K256.json` |
| 4096x4096x4096 | v0_naive | 405.2236 | 339.2 | 0.5 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v1_coalesced | 307.1506 | 447.5 | 0.7 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v2_tiled | 96.1345 | 1429.7 | 2.1 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v3_tiled_vectorized | 98.0207 | 1402.1 | 2.1 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v4_coarsened | 45.8339 | 2998.6 | 4.4 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v5_regtile | 25.2631 | 5440.3 | 8.0 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v6_vec_transposed | 22.4809 | 6113.6 | 9.0 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | cublas | 13.6699 | 10054.1 | 14.7 | `bench_M4096_N4096_K4096.json` |
| 4096x512x1024 | v0_naive | 5.2393 | 819.8 | 5.2 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v1_coalesced | 5.7410 | 748.1 | 4.7 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v2_tiled | 2.2510 | 1908.0 | 12.1 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v3_tiled_vectorized | 2.4215 | 1773.7 | 11.3 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v4_coarsened | 1.3183 | 3257.9 | 20.7 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v5_regtile | 0.7750 | 5542.1 | 35.2 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v6_vec_transposed | 0.5885 | 7298.4 | 46.3 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | cublas | 0.4493 | 9559.0 | 60.7 | `bench_M4096_N512_K1024.json` |
| 512x1024x1024 | v0_naive | 1.6491 | 651.1 | 5.1 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v1_coalesced | 1.6577 | 647.7 | 5.1 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v2_tiled | 0.6071 | 1768.7 | 13.8 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v3_tiled_vectorized | 0.6294 | 1705.9 | 13.3 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v4_coarsened | 0.2910 | 3689.7 | 28.8 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v5_regtile | 0.1953 | 5498.0 | 43.0 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v6_vec_transposed | 0.1465 | 7329.5 | 57.3 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | cublas | 0.1124 | 9548.8 | 74.6 | `bench_M512_N1024_K1024.json` |
| 512x4096x1024 | v0_naive | 5.0075 | 857.7 | 5.4 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v1_coalesced | 5.6741 | 756.9 | 4.8 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v2_tiled | 2.2637 | 1897.3 | 12.0 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v3_tiled_vectorized | 2.4078 | 1783.7 | 11.3 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v4_coarsened | 1.3141 | 3268.4 | 20.7 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v5_regtile | 0.7714 | 5568.0 | 35.3 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v6_vec_transposed | 0.5910 | 7266.8 | 46.1 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | cublas | 0.4559 | 9420.1 | 59.8 | `bench_M512_N4096_K1024.json` |
| 512x512x512 | v0_naive | 0.3410 | 787.1 | 9.2 | `bench_M512_N512_K512.json` |
| 512x512x512 | v1_coalesced | 0.2606 | 1029.9 | 12.1 | `bench_M512_N512_K512.json` |
| 512x512x512 | v2_tiled | 0.1818 | 1476.3 | 17.3 | `bench_M512_N512_K512.json` |
| 512x512x512 | v3_tiled_vectorized | 0.1781 | 1507.4 | 17.7 | `bench_M512_N512_K512.json` |
| 512x512x512 | v4_coarsened | 0.0828 | 3240.1 | 38.0 | `bench_M512_N512_K512.json` |
| 512x512x512 | v5_regtile | 0.0952 | 2819.7 | 33.0 | `bench_M512_N512_K512.json` |
| 512x512x512 | v6_vec_transposed | 0.0796 | 3374.3 | 39.5 | `bench_M512_N512_K512.json` |
| 512x512x512 | cublas | 0.0397 | 6759.6 | 79.2 | `bench_M512_N512_K512.json` |
<!-- RESULTS-END -->

Plots: `results/plot_gflops.png` (median GFLOPS vs shape per kernel, generated by
`python3 scripts/plot_results.py` from the bench JSONs — re-run the script after
any new bench run and commit the figure).

## Key lessons (measured — see `docs/WRITEUP.md` for evidence)

- Occupancy + work-per-thread beat wider loads: `v4` (256-thread blocks, 2×2
  outputs/thread) is 1.8× faster than `v2` at 1024³ (0.64 vs 1.12ms); `v3`'s
  float4 loads never beat `v2` beyond noise (v3 1.31 vs v2 1.12ms at 1024³) —
  a recorded negative result.
- Bigger tiles win big problems, starve small ones: `v5` (128-tile, 8×8/thread)
  is 1.5× faster than `v4` at 1024³ (0.425 vs 0.64ms) but loses at 512³ and below
  (0.32× at 256³ — 4 blocks can't fill the GPU).
- Layout + vectors stack on top: `v6` (float4 fills, transposed A tile) is 1.4×
  faster than `v5` on every shape including 256³ (0.306 vs 0.425ms at 1024³,
  67% of cuBLAS) —
  shared-memory layout was second-order, but not zero.
- The vendor baseline earned its place twice: it caught our wrong-math OP_T
  emulation (fixed to OP_N, validated value-by-value), and it leads the best
  custom kernel 1.5× at 1024³ (10456 vs 7026 GFLOPS).
- Protocol before speed: `cudaEvent` kernel-only timing, warmup plus timed iters,
  median/min/p95, identical inputs, JSON evidence with environment metadata,
  repeat runs with a scripted >5% drift check (`scripts/check_repeat.py`).
- Correctness-first ladder: each version is its own file with bounds-checked
  arbitrary sizes, so `git bisect` / per-version benching stays reproducible.
- ML link, measured (`ml/transformer_gemm.py`, torch 2.14.1+cu130): FP32 eager
  `torch.matmul` on transformer GEMMs (512/2048 tokens × 4096, FFN 11008)
  matches the cuBLAS harness within 9% (framework overhead ≈ noise); v6 trails
  eager 1.5–1.8×; `torch.compile` NOT RUN (no native Windows Triton).
  Evidence: `results/ml_transformer.json`, `results/ml_M*_N*_K*.json`.

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
- Laptop GPU (RTX 5070 Laptop, 75 W power cap, display-attached WDDM): clocks
  swing run to run (observed SM boost 1447–2730 MHz across bench runs); medians
  plus repeats plus drift adjudication carry the load this creates. Numbers
  compare runs on this machine only.
- FP32 peak (13.13 TFLOPS) is estimated from the driver-reported 1425 MHz SM
  clock, but measured boost reaches 2730 MHz — the estimate is likely
  understated, so achieved-% claims use it as a labeled estimate, not a roof.
- No `nsys` on this machine: kernel-time (`cudaEvent`) only, no end-to-end
  traces; operator-level GFLOPS is not end-to-end latency (see `docs/WRITEUP.md`
  §16).


## References

See `docs/REFERENCES.md` (programming guide, best-practices guide, matrixMul/tiled
pattern, cuBLAS, Nsight Compute, Blackwell toolkit notes). Adapted ideas are also
cited in code comments.
