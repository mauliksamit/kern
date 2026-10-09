# LOG.md — engineering log (plain language; one note per optimization step)

## 2026-10-02 — Scaffold (no GPU toolchain on this machine)
- Bottleneck: none yet; repo did not exist. `nvidia-smi` shows an RTX 5070
  (driver 576.76, CUDA 12.9) but `nvcc`/`ncu`/`nsys`/`compute-sanitizer` are all
  NOT FOUND and no CUDA toolkit directory exists. So GPU code cannot be compiled
  or timed here.
- Change: created the full scaffold: CMake (CPU-only fallback
  + CUDA when present, `-O3 -lineinfo`, `sm_120` default with override),
  `kern/cpu` reference (double accumulation), kernels v0 naive / v1 coalesced /
  v2 tiled / v3 tiled-vectorized (each its own file, bounds-checked), `bench`
  (cudaEvent kernel-only, 10 warmup / 100 timed, median/min/p95, GFLOPS+GB/s
  defined in code, JSON with git/env/flags/seed), `correctness` (awkward shapes,
  fixed seeds, justified tolerance), scripts (tables, claims audit, sanitizer,
  ncu, arch detect), docs (ENVIRONMENT/WRITEUP/NEGATIVE/REFERENCES), CI
  (CPU-only), LICENSE, .gitignore, .clang-format.
- Metric: no GPU metric (NOT RUN). CPU gates verified below; GPU evidence pending
  the reproduce commands in `docs/ENVIRONMENT.md`.
- Next: install CUDA toolkit 12.8+ on a bench machine, run correctness +
  sanitizer + `kern_bench --all` twice, then fill README/WRITEUP from `results/`.

## 2026-10-02 — Verification on scaffolding machine (CPU-only)
- `python tests/test_correctness_cpu.py`: ALL PASS (8 shapes incl. size-1, primes,
  non-multiples, very large/small magnitudes, seed 1234).
- `cmake` + `cmake --build` + `ctest`: PASS in both `build/` and fresh `build-cold/`
  (cold-clone repro, MinGW Makefiles + g++ 16.1.0; default NMake generator fails
  without `nmake`, documented in README). CPU-only targets as expected (no `nvcc`).
- `scripts/claims_audit.py`: PASSED (0 traced values; README contains no untraced
  measurement claims). `scripts/generate_tables.py --check`: fresh.
- GPU gates (`kern_correctness`, `kern_bench`, sanitizer, `ncu`): NOT RUN here.
  First bench run must also confirm the `sm_120` arch inference via
  `tools/detect_arch.py` and record the outcome in `docs/ENVIRONMENT.md`.

## (append per step; one change per commit, message = hypothesis + change + before/after median + profiler evidence)

## 2026-10-03 — Toolchain installed, code compiled, execution blocked by policy
- Bottleneck: no CUDA toolkit and no MSVC host compiler on the machine.
- Change: installed CUDA 13.0 (`winget install Nvidia.CUDA --version 13.0`) and
  VS 2022 Build Tools + VCTools workload (`cl` 19.44.35229). Configured with the
  NMake generator inside `VsDevCmd -arch=amd64` plus an explicit
  `-DCMAKE_CUDA_COMPILER=.../nvcc.exe` (CMake's VS-generator path needs the CUDA
  VS-integration, which was absent since VS was installed after CUDA; NMake +
  explicit compiler avoids it). Two real bugs surfaced by the real compiler and
  fixed: (1) `src/bench.cu` used `cudaDeviceProp::clockRate`, removed in CUDA 13 —
  clocks now come from `cuDeviceGetAttribute(CLOCK_RATE / MEMORY_CLOCK_RATE)` and
  are stored raw (kHz) in the JSON; (2) `v3` used `__launch_bounds__(1024, 2)` and
  ptxas rejected the min-blocks hint as exceeding SM residency — now `(1024, 1)`.
- Metric: full build succeeds (`kern_kernels`, `kern_bench`, `kern_correctness`
  for sm_120). GPU correctness/bench/sanitizer: NOT RUN — Smart App Control
  (`VerifiedAndReputableDesktop`, CI events 3077/3118) blocks the freshly built
  unsigned binaries and no Defender cloud verdict was retrieved, so no reputation
  exception applies. A trivial signed-toolchain hello.exe runs; the block is
  content/reputation-based, not path-based. NOT bypassed (user/org security
  control). Next: turn SAC off or run the bench commands on an unblocked
  CUDA machine; then fill README/WRITEUP from `results/`.

## 2026-10-09 — Phase A audit: real bug in v3 (divergent __syncthreads)
- Bottleneck: none measured yet; this is a correctness fix found by inspection.
  `kernels/v3_tiled_vectorized.cu` chose its float4 fast path with a per-thread
  `(row, col)` alignment test, so threads in one 32x32 block could disagree and
  execute *different* `__syncthreads()` barriers — undefined behavior, can hang.
  Supporting evidence: pre-fix `build/kern_correctness.exe` started (PID live)
  but produced zero output in 120 s; post-fix it exits in <45 s with ALL PASS.
  (Stdout was block-buffered to a file, so the hang location is inferred from
  shape order — first divergence-prone shape is 100x97x63 — not proven.)
- Change: fast-path decision is now block-uniform: tile-base pointers plus
  `K % 4 == 0 && N % 4 == 0` (so every row shares the base 16B alignment).
  Added `<cstdint>`, size_t index math. One isolated change, no protocol change.
- Metric: `kern_correctness --seed 1234`: ALL PASS, worst abs errs 0 (1x1x1) to
  3.24e-05 (512^3); large-mag case passes on relative tolerance as designed.
  Log: `results/correctness_seed1234.json`.
- Environment update: driver is now 617.14 / CUDA UMD 13.4 (was 576.76/12.9);
  `nvcc` 13.0.88, `ncu` + `compute-sanitizer` 2025.3.1.0 present, `nsys` absent,
  `cl`/`nmake` via VS 2022 Build Tools (14.44) + VsDevCmd only. `hello.exe`
  (real CUDA: malloc+kernel+memcpy) prints 42 — basic GPU execution works here,
  so the SAC block no longer stops execution; the hang was the v3 bug, not SAC.
- Next: sanitizer (running), then `kern_bench --all` twice + drift check.

## 2026-10-09 — Correctness gate passes on RTX 5070 (sm_120, Release)
- Bottleneck: unvalidated kernels (all GPU claims were NOT RUN).
- Change: none to code; ran `./build/kern_correctness --seed 1234` after the v3
  fix + `cmake --build build` (NMake, Release, sm_120 — clean).
- Metric: ALL PASS (bad=0) over 10 standard + 2 magnitude shapes x 4 kernels =
  48 checks. Worst abs err scales with K as expected (2.38e-07 at K=16 up to
  3.24e-05 at K=512). CPU gates re-verified: `test_correctness_cpu.py` ALL PASS,
  `claims_audit.py` PASSED (0 traced values), `generate_tables.py --check` fresh.
- Next: sanitizer logs, then benchmarks.

## 2026-10-09 — cuBLAS baseline was wrong math; fixed + validated (two bugs)
- Bottleneck: the vendor baseline — the honest target — computed the wrong
  operation, so every "vs cuBLAS" comparison would have been meaningless.
- Bug 1 (found by bench abort): `time_cublas` used OP_T+swapped buffers, whose
  transposes cancel to B*A instead of C. On shape M=1024,N=1024,K=512 cuBLAS
  refused the call ("parameter 10 (LDB) illegal") and the old `exit(1)` killed
  the whole `--all` run, losing that shape's custom-kernel timings too.
  Fix: correct recipe is swapped buffers with OP_N,OP_N
  (`cublasSgemm(h,N,N,N,M,K,alpha,dB,N,dA,K,beta,dC,N)` = C^T = B^T*A^T),
  exact for ALL shapes with no copies; plus failure-tolerant JSON
  (`"failed"` flag, skipped by `generate_tables.py`) instead of `exit(1)`.
- Bug 2 (found by the new gate): host readback indexed the N x M col-major C^T
  as row-major (`hCt[j*M+i]`); only [0,0] passed. Fix: `(C^T)[j][i]` lives at
  `i*N+j` in col-major. Pattern diagnosed from output, not guessed.
- Change: `src/bench.cu` (baseline rewrite + `failed` flag), `tests/…`
  (new `check_cublas` on all 12 shapes vs CPU ref), `CMakeLists.txt` (link
  `CUDA::cublas` into `kern_correctness`), `generate_tables.py` (skip failed).
- Metric: `kern_correctness --seed 1234` ALL PASS (bad=0): 48 custom-kernel +
  12 cuBLAS checks. `results/correctness_seed1234.json` rewritten.
  Lesson: a baseline needs its own correctness check — timing the wrong math
  at the right speed is the most expensive kind of wrong. Run-1 bench numbers
  (pre-fix baseline) are discarded; fresh `--all` x2 required.
- Next: sanitizer re-run on final binaries (running), then bench x2 + drift.

## 2026-10-09 — Phase B done: benchmarks + drift adjudication (runs A-E)
- Bottleneck: no trustworthy numbers (run-1 pre-fix data discarded: wrong-math
  baseline + aborted 5th shape).
- Change: none to protocol; `kern_bench --all --seed 0` twice (D, E) on the
  6-kernel binary + scripted drift check (`scripts/check_repeat.py`, new) +
  single-shape run C for the drifting 1024^3 cell. Run E kept as reported run.
- Metric (run E medians, `results/bench_*.json`): 1024^3: v0 2.4838 / v1 2.2586 /
  v2 1.1672 / v3 1.2242 / v4 0.6212 / cublas 0.2259 ms. v4 = 3457 GFLOPS (26% of
  13.13 TFLOPS est. peak: 36 SMs x 128 x 2 x 1.425 GHz); cuBLAS = 9508 (72%).
  Drift: 29/30 cells within 5%. Two marginals investigated, not hidden: (1)
  A-vs-B 1024^3 v2 9.50% — run C (1.1352) agreed with A (1.1175), B (1.2236)
  the outlier, distributions overlapping (mins 1.0956/1.1294/1.0981); (2) D-vs-E
  K512 v4 5.42% with identical mins (0.2601/0.2613) — WDDM noise on a 0.27 ms
  kernel. Both disclosed in `docs/WRITEUP.md` section 10.
- `ncu` attempted (SpeedOfLight, 256^3): `ERR_NVGPUCTRPERM` — counters blocked
  on this WDDM device; enabling needs admin action (out of scope). Attempt log
  saved to `results/ncu/ncu_smoke_M256_attempt.log`; profiler claims stay NOT
  RUN. Tables regenerated (30 rows), claims audit passes (180).
- Next: Phase C v4 was already built/measured (see next entry); remaining:
  ctest, git init + first commit, final assessment.

## 2026-10-09 — Phase C: v4 coarsening 1.9-2.2x over v2 (hypothesis confirmed)
- Bottleneck (evidence): v2/v3 1024-thread blocks cap residency at ~1 block/SM;
  v3's wider loads gained nothing (see NEGATIVE_RESULTS), pointing at occupancy,
  not bandwidth, as binding.
- Change (single, isolated): `kernels/v4_coarsened.cu` — same 32x32 tile +
  shared algorithm as v2, but 16x16 threads x 2x2 outputs each (256 threads,
  cooperative 4-element fill, 4 guarded stores). Registered in kernels.h,
  CMakeLists, bench (6th entry), correctness gate. No protocol change.
- Metric: correctness ALL PASS (bad=0, 72 checks incl. 12 cuBLAS); bench D/E:
  v4 beats v2 1.55x (256^3), 2.2x (512^3), 1.9x (1024^3), ~2.2x (rects).
  1024^3: 0.6394/0.6212 vs v2 1.1930/1.1672. cuBLAS still leads 2.7x (0.2259).
  Sanitizer re-run on final binaries (running at log time).
- Next: docs refresh (README/WRITEUP/NEGATIVE — done this pass),
  plot script, ctest, git init + commit, Phase D/E assessment.

## 2026-10-09 — First commit; evidence-to-revision mapping; session closeout
- `git init` + first commit `ba04ec7` (working tree clean at commit time).
  `results/bench_*.json` record `"git_commit": "unknown"` because no repo
  existed when they were measured; mapping (verified, tree clean): all GPU
  evidence in this commit (bench runs D/E + run C, correctness seed-1234,
  sanitizer batch 3, ncu attempt log, plot) was produced from exactly the
  committed sources. Future runs will stamp the hash automatically.
- ctest: 3/3 pass (cpu reference, gpu_correctness 72/72, claims_audit 185
  traced). CPU gates + roofline self-check + table freshness re-verified
  pre-commit. `.gitignore` += `*.exe` (root hello.exe stays local, untracked).
- Residual gaps (not hidden): `ncu` counters blocked (NOT RUN evidence kept);
  `nsys` absent; torch install deferred (Phase D analytical only via
  `scripts/roofline.py`); CI has no clang-format step;
  percentages in README derive from traced values (audit understands 100*a/b).
- Next: push to GitHub (creates remote; CI runs CPU checks),
  optional torch `ml/` experiment per WRITEUP section 13, optional ncu run on
  a counter-permitted machine per `scripts/run_ncu.sh`.

## 2026-10-09 — Independent audit: no v2/v4 swap; format + CI + rounding fixes
- Audit premise (v2=0.62 / v4=1.17 swap at 1024^3): NOT FOUND in any file.
  Recomputed all 30 bench cells from raw `iters` arrays: medians exact,
  GFLOPS/GB/s recompute from dims (2MNK/t, ideal-bytes/t). Raw 1024^3:
  v2 median 1.167232 ms / 1839.81 GFLOPS, v4 median 0.621248 ms / 3456.72
  GFLOPS (v4/v2 = 1.88x). README table (generated), WRITEUP sections 7/9,
  LOG, plot all attribute v2=1.17 / v4=0.62 consistently.
  The reported swap exists nowhere in the repo; claim stands as written.

- Confirmed issue 2 (fixed): `near()` comment claimed sqrt(K) abs_tol scaling
  that the code (intentional `*= 1.0f` no-op anchor) does not do — comment now
  describes actual behavior. No test change (all pass with margin).
- Confirmed gap 3 (fixed): the CI was meant to check formatting; ci.yml lacked it.
  Added pinned `clang-format==23.1.3` dry-run step (same version run locally);
  reformatted 11 source files (token-diff: include regrouping only, verified
  neutral), rebuilt, ctest 3/3 passes. Remote CI result still pending
  (cannot claim passed until observed).
- Verification: fresh `git clone` to temp dir + full NMake/CUDA configure +
  build = COLD-BUILD-OK; cold ctest 3/3 (incl. gpu_correctness ALL PASS).
  Sanitizer batch re-run on formatted binaries: 4/4 clean (0 errors/hazards).
- Evidence notes: correctness JSON stays minimal (per-case lines in sanitizer
  logs: 73 PASS / 0 FAIL each); bench JSON `git_commit: unknown` mapped to
  commits in this LOG entry of 2026-10-09 (tree was clean); no remote
  configured — no push destination yet; PyTorch work deferred
  (roofline analysis only, no new deps).

## 2026-10-09 — Phase 0: evidence base (metadata + doc cleanup)
- Bottleneck: provenance gaps — `git_commit: unknown`, descriptive flags,
  no driver/clocks in JSON; PEDANTIC math unset; stale scaffolding prose.
- Change: bench JSON stamps commit/driver/compile-flags/math-mode/env
  samples/peak-note/max-observed-clock; CMake bakes flags; correctness pins
  PEDANTIC; docs reconciled (72 checks verified vs 72 PASS lines), WRITEUP
  1–14 renumbered, archive/move done. Commits 182f8e3 (metadata), 6012bce
  (docs). Smoke 256^3: commit stamped, PEDANTIC verified, obs 1845MHz vs
  reported 1425MHz — peak estimate likely understated, labeled as such.
- Metric: pending full re-run (correctness/sanitizer/bench×2 next).
- Next: Phase 0 evidence commit, then Phase 1 ncu.

## 2026-10-09 — Phase 0 evidence: reruns under contention, reported run G
- Bottleneck: new JSON metadata needed validation; then the session turned
  disturbed — F-vs-G drift flagged 8 cells (worst 46% on 1024^3 v2).
- Change: correctness ALL PASS (bad=0) + sanitizer 4/4 clean on metadata
  binaries; bench --all runs F, G, H, I (seed 0); runs preserved aside
  (runF/runG/runH/runI in temp); reported run G restored to results/.
- Metric: env samples caught non-stationary clocks (F: 1627->1447 MHz
  falling; G: 1627->2377 MHz rising, 71 W near cap) — machine state, not
  kernel behavior. 4-run + history adjudication: G has fewest
  history-disagreeing cells. Run G 1024^3 medians: v0 2.4845 / v1 2.2594 /
  v2 1.1263 / v3 1.2700 / v4 0.6481 (3314 GFLOPS, 25% of est. peak) /
  cuBLAS 0.2410 (8910 GFLOPS, 68%). Ranges: v4/v2 1.5-2.2x across shapes;
  cuBLAS/v4 2.0-2.7x at large shapes. v3 never beats v2 beyond noise
  (best: 4% at 512^3). All prose refreshed to G; audit passes (185 traced).
- Process failure, owned: run E's JSONs were overwritten by run F before
  being copied aside. E values survive in prior LOG/WRITEUP prose only.
  Rule from here: copy each run aside BEFORE the next run starts.
- ncu: counters STILL blocked after they were enabled (even --query-metrics
  fails; logs in results/ncu/). Skipping to Phase 2, profiler NOT RUN.
  Device note: ncu reports GB206 (earlier GB205 inference wrong; sm_120
  flag unaffected — builds and runs).
- Next: Phase 2 v5_regtile (BM=BN=128, BK=8, 8x8 micro-tile, 256 threads).
