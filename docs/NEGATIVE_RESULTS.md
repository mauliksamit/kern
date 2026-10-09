# Negative results (keep every miss)

Status 2026-10-09: GPU experiments HAVE been run on this machine (RTX 5070,
sm_120; correctness 72/72, bench runs D/E + run C, sanitizer 4/4 clean — see
`LOG.md`, `results/`). Entries 1–3 below were **design-stage rejections**
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
   Same verdict across sessions D/E/F/G/H/I: no genuine gain — cooperative
   float4 fill + fast-path branch overhead at the 1024-thread occupancy wall.
   v4 (same tile, coarsened threads) gained 1.5-2.2x, confirming the wall,
   not load width, was binding. `ncu` transaction counters NOT RUN (blocked,
   see `results/ncu/`). v3 file retained; do not build on it without evidence.
