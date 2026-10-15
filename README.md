# kern — SGEMM optimization ladder (CUDA)

What: single-precision matrix multiply `C = A·B` (row-major) optimized step by step —
`v0` naive → `v1` coalesced → `v2` shared-memory tiled → `v3` tiled + vectorized →
`v4` thread-coarsened (2×2 outputs/thread) → `v5` register-blocked (128-tile,
8×8/thread) → `v6` vectorized + transposed — plus a cuBLAS vendor baseline.
Why: a small, reproducible study of GPU performance work where every
claim traces to a committed log file.

Status: **measured on RTX 5070 (sm_120), 2026-10-09.** Correctness gate ALL PASS
(160 checks: 20 shapes × (7 kernels + cuBLAS)) and Compute Sanitizer
clean (memcheck/racecheck/synccheck/initcheck), both re-run 2026-10-15 on the
default strict-IEEE build (`KERN_FAST_MATH=OFF`, no `--use_fast_math`; commit,
flags and binary SHA-256 in each `results/sanitizer/*.log` header), bench run twice
with drift check + adjudicating runs. Speed claims rest on `cudaEvent`
timing; mechanism claims rest on a per-kernel Nsight Compute profile at 1024³
(v2/v4/v5/v6/cuBLAS, 2026-10-15, `results/ncu/ladder_M1024_N1024_K1024/summary.md`), which
overturned two of our own earlier explanations (see Key lessons).

## Hardware used

- Bench + scaffolding machine: NVIDIA GeForce RTX 5070 (sm_120, Blackwell),
  driver 617.14, AMD Ryzen AI 7 350, Windows 64-bit, WDDM (display-attached —
  expect scheduling noise; hence medians, repeats, drift checks). CUDA 13.0 +
  MSVC 19.44, CMake 4.4.3, `ncu`/`compute-sanitizer` 2025.3.1.0, no `nsys`.
  Full record: `docs/ENVIRONMENT.md`.
- Headline result (1024³, medians, reported run V; runs U + O/P/N + J/K/L/M/F/G/H/I disclosed
  in `docs/WRITEUP.md` §13):
  `v0` 2.64ms → `v1` 2.29ms → `v2` 1.13ms → `v3` 1.31ms → `v4` 0.64ms →
  `v5` 0.425ms → **`v6` 0.310ms** (6934 GFLOPS, 69% of cuBLAS) →
  cuBLAS 0.213ms (10093 GFLOPS). Details + caveats:
  `docs/WRITEUP.md`.

## Build, run, reproduce (one command each)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKERN_CUDA_ARCH=sm_120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

(`sm_120` matches the RTX 5070 inference; override per GPU, e.g. `sm_80` for A100.
Needs CUDA 12.8 or newer for 50-series (verified here with CUDA 13.0 + MSVC 19.44).
Fast math is off by default; all committed numbers use strict IEEE FP unless a
bench JSON's `compile_flags` says otherwise (`-DKERN_FAST_MATH=ON` to opt in).
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
| 1024x1024x1024 | v0_naive | 2.6367 | 814.4 | 4.8 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v1_coalesced | 2.2912 | 937.3 | 5.5 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v2_tiled | 1.1295 | 1901.3 | 11.1 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v3_tiled_vectorized | 1.3083 | 1641.4 | 9.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v4_coarsened | 0.6397 | 3357.1 | 19.7 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v5_regtile | 0.4248 | 5055.3 | 29.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | v6_vec_transposed | 0.3097 | 6934.2 | 40.6 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x1024 | cublas | 0.2128 | 10093.1 | 59.1 | `bench_M1024_N1024_K1024.json` |
| 1024x1024x512 | v0_naive | 1.6134 | 665.5 | 5.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v1_coalesced | 1.1591 | 926.3 | 7.2 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v2_tiled | 0.5999 | 1790.0 | 14.0 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v3_tiled_vectorized | 0.5962 | 1801.1 | 14.1 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v4_coarsened | 0.2756 | 3896.2 | 30.4 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v5_regtile | 0.2109 | 5090.2 | 39.8 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | v6_vec_transposed | 0.1685 | 6370.7 | 49.8 | `bench_M1024_N1024_K512.json` |
| 1024x1024x512 | cublas | 0.1112 | 9656.0 | 75.4 | `bench_M1024_N1024_K512.json` |
| 2048x2048x2048 | v0_naive | 23.8602 | 720.0 | 2.1 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v1_coalesced | 22.2106 | 773.5 | 2.3 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v2_tiled | 10.4853 | 1638.5 | 4.8 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v3_tiled_vectorized | 11.0761 | 1551.1 | 4.5 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v4_coarsened | 4.7135 | 3644.8 | 10.7 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v5_regtile | 2.7826 | 6174.0 | 18.1 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | v6_vec_transposed | 2.2613 | 7597.3 | 22.3 | `bench_M2048_N2048_K2048.json` |
| 2048x2048x2048 | cublas | 1.5808 | 10868.1 | 31.8 | `bench_M2048_N2048_K2048.json` |
| 256x256x256 | v0_naive | 0.0725 | 462.5 | 10.8 | `bench_M256_N256_K256.json` |
| 256x256x256 | v1_coalesced | 0.0344 | 974.5 | 22.8 | `bench_M256_N256_K256.json` |
| 256x256x256 | v2_tiled | 0.0281 | 1194.3 | 28.0 | `bench_M256_N256_K256.json` |
| 256x256x256 | v3_tiled_vectorized | 0.0282 | 1188.9 | 27.9 | `bench_M256_N256_K256.json` |
| 256x256x256 | v4_coarsened | 0.0176 | 1903.0 | 44.6 | `bench_M256_N256_K256.json` |
| 256x256x256 | v5_regtile | 0.0573 | 585.8 | 13.7 | `bench_M256_N256_K256.json` |
| 256x256x256 | v6_vec_transposed | 0.0472 | 710.4 | 16.6 | `bench_M256_N256_K256.json` |
| 256x256x256 | cublas | 0.0188 | 1783.3 | 41.8 | `bench_M256_N256_K256.json` |
| 4096x4096x4096 | v0_naive | 408.7477 | 336.2 | 0.5 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v1_coalesced | 314.9752 | 436.3 | 0.6 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v2_tiled | 96.4199 | 1425.4 | 2.1 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v3_tiled_vectorized | 98.6119 | 1393.7 | 2.0 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v4_coarsened | 45.8335 | 2998.7 | 4.4 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v5_regtile | 25.2083 | 5452.1 | 8.0 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | v6_vec_transposed | 22.6764 | 6060.9 | 8.9 | `bench_M4096_N4096_K4096.json` |
| 4096x4096x4096 | cublas | 13.5843 | 10117.5 | 14.8 | `bench_M4096_N4096_K4096.json` |
| 4096x512x1024 | v0_naive | 5.2595 | 816.6 | 5.2 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v1_coalesced | 5.8387 | 735.6 | 4.7 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v2_tiled | 2.2587 | 1901.5 | 12.1 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v3_tiled_vectorized | 2.4696 | 1739.1 | 11.0 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v4_coarsened | 1.3018 | 3299.3 | 20.9 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v5_regtile | 0.7723 | 5561.6 | 35.3 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | v6_vec_transposed | 0.5849 | 7342.7 | 46.6 | `bench_M4096_N512_K1024.json` |
| 4096x512x1024 | cublas | 0.4115 | 10438.5 | 66.3 | `bench_M4096_N512_K1024.json` |
| 512x1024x1024 | v0_naive | 1.2835 | 836.6 | 6.5 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v1_coalesced | 1.1547 | 929.9 | 7.3 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v2_tiled | 0.6266 | 1713.5 | 13.4 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v3_tiled_vectorized | 0.6154 | 1744.7 | 13.6 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v4_coarsened | 0.2804 | 3829.5 | 29.9 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v5_regtile | 0.1885 | 5695.9 | 44.5 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | v6_vec_transposed | 0.1487 | 7219.1 | 56.4 | `bench_M512_N1024_K1024.json` |
| 512x1024x1024 | cublas | 0.1125 | 9543.4 | 74.6 | `bench_M512_N1024_K1024.json` |
| 512x4096x1024 | v0_naive | 5.1474 | 834.4 | 5.3 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v1_coalesced | 5.6516 | 760.0 | 4.8 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v2_tiled | 2.2481 | 1910.5 | 12.1 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v3_tiled_vectorized | 2.4204 | 1774.5 | 11.3 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v4_coarsened | 1.2828 | 3348.0 | 21.3 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v5_regtile | 0.7691 | 5584.5 | 35.4 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | v6_vec_transposed | 0.5955 | 7212.1 | 45.8 | `bench_M512_N4096_K1024.json` |
| 512x4096x1024 | cublas | 0.4626 | 9285.2 | 58.9 | `bench_M512_N4096_K1024.json` |
| 512x512x512 | v0_naive | 0.3394 | 790.9 | 9.3 | `bench_M512_N512_K512.json` |
| 512x512x512 | v1_coalesced | 0.2621 | 1024.0 | 12.0 | `bench_M512_N512_K512.json` |
| 512x512x512 | v2_tiled | 0.1813 | 1480.5 | 17.4 | `bench_M512_N512_K512.json` |
| 512x512x512 | v3_tiled_vectorized | 0.1784 | 1504.4 | 17.6 | `bench_M512_N512_K512.json` |
| 512x512x512 | v4_coarsened | 0.0817 | 3287.1 | 38.5 | `bench_M512_N512_K512.json` |
| 512x512x512 | v5_regtile | 0.0946 | 2836.9 | 33.2 | `bench_M512_N512_K512.json` |
| 512x512x512 | v6_vec_transposed | 0.0793 | 3386.6 | 39.7 | `bench_M512_N512_K512.json` |
| 512x512x512 | cublas | 0.0395 | 6792.4 | 79.6 | `bench_M512_N512_K512.json` |
<!-- RESULTS-END -->

Plots: `results/plot_gflops.png` (median GFLOPS vs shape per kernel, generated by
`python3 scripts/plot_results.py` from the bench JSONs — re-run the script after
any new bench run and commit the figure).

## Key lessons (measured — see `docs/WRITEUP.md` for evidence)

- Work per instruction, not occupancy: `v4` (256-thread blocks, 2×2
  outputs/thread) is 1.8× faster than `v2` at 1024³ (0.64 vs 1.13ms). We first
  credited occupancy; ncu shows both at 66.67% theoretical occupancy, while
  shared-load instructions fall 70% (41,943,040 → 12,582,912) and the FMA pipe
  goes from 8.65% to 20.92% active. `v3`'s float4 loads never beat `v2` beyond
  noise (v3 1.31 vs v2 1.13ms at 1024³) — a recorded negative result.
- Bigger tiles win big problems, starve small ones: `v5` (128-tile, 8×8/thread)
  is 1.5× faster than `v4` at 1024³ (0.425 vs 0.64ms) but loses at 512³ and below
  (0.31× at 256³ — 4 blocks can't fill the GPU).
- The fill, not the layout: `v6` (float4 fills, transposed A tile) is 1.4×
  faster than `v5` on every shape including 256³ (0.310 vs 0.425ms at 1024³,
  69% of cuBLAS). We designed the transposed tile to remove A-fragment bank
  conflicts; ncu shows v5 and v6 with identical shared-load conflicts
  (4,194,304 each). The measured change is in the fill: shared-store
  conflicts 1,588,225 → 278,453 and 8% fewer warp instructions.
- The vendor baseline earned its place twice: it caught our wrong-math OP_T
  emulation (fixed to OP_N, validated value-by-value), and it leads the best
  custom kernel 1.5× at 1024³ (10093 vs 6934 GFLOPS). The cuBLAS kernel here
  is a CUTLASS SIMT FP32 kernel (no tensor cores) doing the same FFMA count as
  v6; its FMA pipe is 60.42% active vs v6's 33.91%.
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
- Profiler scope: `ncu` counters were blocked (`ERR_NVGPUCTRPERM`) until
  2026-10-15. The counter profile covers one shape (1024³) and five kernels
  (v2, v4, v5, v6, cuBLAS), one launch each, with clocks unlocked
  (`--clock-control none`); v0/v1/v3 and other shapes are NOT RUN under ncu.
  DRAM byte counts did not reproduce across two runs and are not used.
- `v2`/`v3` use 32x32 thread blocks (1024 threads, the architectural max):
  one block per SM, which ncu shows is still 66.67% theoretical occupancy
  (same as `v4`); see `docs/WRITEUP.md` §10.
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

Verified numbers with their source files: `docs/FACTS.md` (generated by
`scripts/facts.py` from `results/`).

See `docs/REFERENCES.md` (programming guide, best-practices guide, matrixMul/tiled
pattern, cuBLAS, Nsight Compute, Blackwell toolkit notes). Adapted ideas are also
cited in code comments.
