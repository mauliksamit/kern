# Negative results (keep every miss)

Status 2026-10-09: GPU experiments HAVE been run on this machine (RTX 5070,
sm_120; correctness 160/160, bench runs with drift adjudication, sanitizer
4/4 clean on the strict-IEEE build - see `LOG.md`, `results/`). Entries 1–3 below were **design-stage rejections**
(never run, kept as decisions); entry 5 is a **measured** miss. Append new
measured misses here with the reason.

1. **Larger 64x64 thread-block tiles** — REJECTED at design time, NOT RUN.
   Reason: 64x64 = 4096 threads exceeds the 1024 threads/block limit; would need
   coarsening (each thread covers 4+ elements). Parked as a possible v4 with
   2D thread coarsening instead. Do not claim any effect without measurement.
2. **Double-precision accumulation inside GPU kernels** — REJECTED at design time.
   Reason: reference accumulates in double on CPU (ground truth), but GPU kernels
   deliberately accumulate in float so the benchmark reflects real SGEMM throughput.
   Mixed-precision comparison is a separate study.
3. **Timing H2D/D2H copies inside the kernel timer** — REJECTED (protocol).
   Reason: the benchmark protocol forbids including host-device copies unless that is
   what is measured. Copies stay outside the `cudaEvent` region in `src/bench.cu`.
4. **(placeholder) v1 vs v0 if v1 does not help** — If measurement shows v1
   median within noise of v0, record medians + p95 here, keep v1 file for
   reproducibility, and note the hypothesis failure in `LOG.md` + `docs/WRITEUP.md`.
   UPDATE 2026-10-09: v1 beats v0 everywhere (2.04x at 256^3, 1.08x at 1024^3),
   so this placeholder did not trigger. The hypothesis held; magnitude varies
   by shape (mapping overhead matters most where kernels are short).
5. **(placeholder) v3 vectorization fallback** — If aligned `float4` path shows no
   gain over v2 on the bench GPU, record `ncu` DRAM-throughput metrics here and
   revert/park.
   UPDATE 2026-10-09, re-verified on reported run G (hypothesis failed again,
   kept reproducible): v3 vs v2 medians (`results/bench_*.json`): 256^3:
   0.0283 vs 0.0275; 512^3: 0.1760 vs 0.1838 (only win, 4%, within noise);
   1024^3: 1.2700 vs 1.1263; K512: 0.5938 vs 0.5923; rect: 0.6098 vs 0.6039.
   Same verdict across sessions D/E/F/G/H/I/J/K/L/M/N: no genuine gain — cooperative
   float4 fill + fast-path branch overhead.
   v4 (same tile, coarsened threads) gained 1.5-2.2x. CORRECTION 2026-10-15:
   this entry used to say v4 confirmed an "occupancy wall"; ncu shows v2 and
   v4 at the same 66.67% theoretical occupancy, so the binding constraint was
   shared-load instruction count, not occupancy (entry 9). v3 itself is not
   in the ncu profile (NOT RUN). v3 file retained; do not build on it without evidence.
   Re-verified on runs J/K/L/M (reported run M): v3 vs v2 medians 256^3:
   0.0287 vs 0.0277; 512^3: 0.1776 vs 0.1852; 1024^3: 1.2580 vs 1.1277;
   K512: 0.5909 vs 0.5931; rect: 0.6142 vs 0.6090. Same verdict, sixth time.
   Re-verified on run N (9 shapes): v3 slower on 7/9 (worst +17% at 1024^3);
   wins only at 512^3 (2%) and K512 (1%). Same verdict, seventh time.
   Re-verified on runs U/V (reported V): v3 slower on 6/9 (worst +16% at 1024^3);
   wins only at 512^3 (2%), K512 (1%) and rect (2%). Same verdict, eighth time.

6. **v7 warp-level tiling — SKIPPED with reason (not a failure).** v6 gained a
   consistent 1.2–1.4x and sits at 73% of cuBLAS (run M era; 69% on reported
   run V) — the remaining gap is vendor
   territory that warp tiling alone cannot take — and without `ncu` counters
   the warp-level effects would be unverifiable tuning in the dark. Parked.
   UPDATE 2026-10-15: counters now work; "tensor cores" was wrong (cuBLAS's
   1024³ kernel is CUTLASS SIMT FP32), see entry 10. See LOG.md.

7. **TDR skip machinery dormant (not a miss).** `kern_bench --all` carries a
   per-shape v0/v1 skip with SKIPPED rendering (tested with a synthetic JSON).
   No shape triggers it: the slowest single launch on record is v0 at 4096^3
   (405–408 ms across runs N/O/P) — 5x below the ~2 s Windows TDR limit. The
   path stays for any future shape beyond 4096.

8. **torch.compile on native Windows — NOT RUN (environment limitation).**
   `ml/transformer_gemm.py` (torch 2.14.1+cu130, RTX 5070 Laptop) records:
   `TritonMissing: Cannot find a working triton installation. Either the
   package is not installed or it is too old.` Triton has no native Windows
   build; per protocol no WSL, no workarounds attempted. Eager FP32 measured
   instead (9.3–10.4 TFLOPS, within 7% of the cuBLAS harness). See
   `results/ml_transformer.json` and LOG.md.

9. **v4 "occupancy" explanation — REFUTED by ncu (2026-10-15).** The speedup
   (1.8x at 1024³) stands; the mechanism we wrote down did not. Theoretical
   occupancy v2 = v4 = 66.67% (achieved 66.32% vs 63.61%). Measured cause:
   shared-load instructions 41,943,040 → 12,582,912, FMA pipe 8.65% → 20.92%
   active. Source: `results/ncu/ladder_M1024_N1024_K1024/summary.md`. Docs corrected in README, WRITEUP §7/§10.

10. **v6 transposed A tile as a bank-conflict fix — NOT SUPPORTED by ncu
    (2026-10-15).** Hypothesis: v5's A-fragment loads are 2-way conflicted and
    storing A transposed makes them broadcasts. Measured: v5 and v6 have
    identical shared-load counts (2,097,152 instructions, 10,485,760
    wavefronts, 4,194,304 conflicts) in two profiling runs. v6's speedup is
    real but the measured differences are in the fill (store conflicts
    1,588,225 → 278,453) and instruction count (42,288,640 → 38,950,912).
    Whether the transpose contributes anything is not isolated: a float4-fill,
    untransposed variant would answer it (NOT RUN). Also corrected: the
    cuBLAS gap is not tensor cores (CUTLASS SIMT FP32 kernel, same FFMA count,
    FMA pipe 60.42% vs 33.91%).
