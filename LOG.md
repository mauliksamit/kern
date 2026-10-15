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

## 2026-10-09 — Phase 2 v5_regtile: 1.5x over v4 at 1024^3, loses small (crossover)
- Bottleneck (evidence): v4's 32^2 tile does 16 MAC per shared float; the
  occupancy win had plateaued (v4 ~3.4 TFLOPS vs cuBLAS ~9.8).
- Change (single): `kernels/v5_regtile.cu` — BM=BN=128, BK=8, 256 threads,
  8x8 register micro-tile (64 C + 8 A + 8 B fragments), scalar shared, no
  swizzle; block-uniform barriers; size_t math; 6 new tile-stress shapes
  (K=1/7/9/129, non-multiples of 128, just-over-128). Idea attribution:
  REFERENCES #8 + code comment (no code copied).
- Metric: correctness ALL PASS (bad=0, 126 checks incl. new shapes);
  sanitizer 4/4 clean; bench J/K (33/35 cells <=5%; two marginals
  adjudicated: 1024^3 v0 8.3% with identical mins, v5 5.9% with K confirmed
  by independent first-look 0.4308). Reported run K 1024^3: v5 0.4298ms /
  4997 GFLOPS (38% est. peak) vs v4 0.6357 (1.48x) vs cuBLAS 0.2190 (2.0x
  lead, 75% peak). Crossover: v5 loses 512^3 (0.87x) and 256^3 (0.33x —
  4 blocks starve the GPU). Claim audit extended to WRITEUP percentages/
  speedups (share + drift forms, analytical-model exemption); passes (215).
- Profiler: ncu still blocked — timing + labeled analysis only, NOT RUN.
- Next: v6_vec_transposed (float4 + transposed A tile) to test layout isolation.

## 2026-10-09 — Phase 2 v6_vec_transposed: 1.3x over v5 everywhere (layout pays)
- Bottleneck (evidence): v5's A-fragment pattern puts both ty-groups on the
  same shared bank (rows 8 apart alias); fill is scalar (4 loads/element-run).
- Change (single): `kernels/v6_vec_transposed.cu` — same BM/BN/BK/TM/TN as v5;
  one float4 per thread in fill (256 float4s = one tile exactly) + A tile
  stored transposed (At[8][128]) so hot fragment loads broadcast; scalar
  fallback into the same layout; tile-level block-uniform decision (one
  barrier site — v3 bug class excluded by construction); 2 new shapes
  (128^3 full-fast, 200^3 mixed). Idea: REFERENCES #8 + comment, no copying.
- Metric: correctness ALL PASS (bad=0, 160 checks: 20 shapes x 8 backends);
  sanitizer 4/4 clean; bench L/M (34/40 cells <=5%; 1024^3 L-burst cells
  adjudicated vs history-matched M). Reported run M 1024^3: v6 0.3053ms /
  7034 GFLOPS (54% est. peak, 73% of cuBLAS) vs v5 0.4096 (1.34x) vs cuBLAS
  0.2214 (1.4x lead). Gains hold on ALL shapes (1.2-1.3x), incl. 256^3
  where v5 starved. Claim audit now scans WRITEUP percentages/speedups
  (share + drift forms, analytical-model exemption); passes (245 traced).
- Profiler: ncu still blocked — timing + labeled analysis only, NOT RUN.
  Without counters the bank-conflict mechanism stays an analytical inference
  from a controlled experiment (only layout/fill changed v5->v6),
  stated as such in WRITEUP.
- Next: Phase 3 broader bench (2048/4096, tall-skinny, wide, TDR skips).

## 2026-10-09 — Phase 3 gatekeeping: 4096^3 bit-exact vs cuBLAS, investigated
- Surprise: `--big` validation shows worst_abs_err=0 EXACTLY at 4096^3-square
  for all 7 kernels, across seeds 1234 and 777 — while 2048^3 (1.6e-4) and
  tall/wide (9e-5) show normal rounding-level diffs vs cuBLAS.
- Investigation (no guessing): verified call args (cublas writes dRef, kernels
  dC — no aliasing), loop/index math, guard balance; instrumented a TEMPORARY
  env-gated spot-dump (reverted after, never committed) showing random nonzero
  inputs and a normal nonzero diff at tall-skinny — the harness demonstrably
  detects differences. First dump misread (tall-skinny overwrote the file;
  gate fixed to exact shape) — owned as a near-miss in method, caught before
  any conclusion.
- Assessment: kernels ran clean (no errors, full 16.7M outputs written) and
  match the vendor baseline to the bit. Best hypothesis: under PEDANTIC math
  at this large-square shape cuBLAS selects a per-element sequential-FMA
  accumulation identical in order to the naive kernels (IEEE-deterministic);
  16.7M-element coincidence by chance is ~impossible, and a harness bug is
  excluded by the nonzero diffs elsewhere. NOT verified (ncu blocked — kernel
  names unobservable), so stated as hypothesis, not finding. The gate's job —
  catching index/bounds/tile bugs, which would show as LARGE errors — is
  unaffected: any such bug would still fail loudly.
- Next: keep the --big gate as is; revisit only if ncu access ever allows
  confirming the cuBLAS kernel selection.

## 2026-10-09 — Phase 3: 9-shape bench, --big gate, TDR dormancy, v7 skipped
- Bottleneck: bench stopped at 1024 (transformer shapes are 4096-wide); no
  big-shape validation; v7 undecided.
- Change (single theme): `--all` runs 9 shapes (2048/4096 squares, tall-skinny
  4096x512x1024, wide 512x4096x1024) with a per-shape TDR skip path (SKIPPED
  rendering tested synthetically); correctness `--big` validates the 4 shapes
  kernel-vs-cuBLAS on GPU (CPU double-ref at 4096^3 would take ~10+ min;
  transitive trust documented); v7 warp-tiling SKIPPED with reason (v6 at 73%
  of cuBLAS, remainder is vendor territory, no counters to validate warps).
- Metric: --big ALL PASS (bad=0); sanitizer 4/4 clean (160/160); bench N/O/P
  (60/72 cells <=5%); reported run N. 4096^3: v6 22.48ms / 6114 GFLOPS (47%
  est. peak) vs cuBLAS 13.67ms (77%); v0 405ms max single launch (5x under
  TDR — skip path dormant, empirically justified). Scaling 1024->4096 (64x
  FLOPs): v6 73x, cuBLAS 67x, v0 163x (cache cliff for no-reuse kernels).
  Tables (72 rows) + plot (9 shapes) regenerated; audit passes (441 traced).
- Next: Phase 4 ml/transformer_gemm.py (venv + CUDA torch or NOT RUN).

## 2026-10-09 — Phase 4: transformer GEMMs measured (torch cu130, compile NOT RUN)
- Bottleneck: ladder-to-workload link was analytical only (roofline estimates).
- Change: `ml/.venv` (Windows python, no elevation) + torch 2.14.1+cu130
  (CUDA on sm_120 verified); `ml/transformer_gemm.py` (5 shapes: attn
  512/2048 x 4096, FFN 11008 up/down; FP32 TF32-off asserted, CUDA events,
  10+100 iters, median/min/p95, float64 numerics); C++ harness benched the
  identical shapes first (results/ml_M*.json); `roofline.py --ml` places them.
- Metric: eager FP32 9.3-10.4 TFLOPS, within 9% of cuBLAS harness (overhead ~=
  noise); v6 trails eager 1.5-1.8x; numerics max rel err <=3.4e-06;
  TF32-allowed row ~2x (separate, labeled); torch.compile NOT RUN
  (TritonMissing, no native Windows Triton, no WSL per protocol) with exact
  error in JSON + NEGATIVE #8. Claims audit extended to ML evidence +
  TFLOPS/1000 rule; passes. Caveat recorded: torch/C++ draws differ (same
  distribution; timing data-independent for dense GEMM).
- Next: Phase 5 docs/finish (writeup per kernel, README, cold clone, audit).

## 2026-10-09 — Phase 5: docs finish, cold clone reproduces within 0.8%
- Bottleneck: per-kernel profiler disclaimers missing; limitations incomplete;
  no cold-clone proof for the final tree.
- Change: one-line "Profiler evidence for this step: NOT RUN (see S13)" in
  S4/S7/S8/S9 (S5/S6/S10/S13 already covered); README limitations += laptop
  power/clock swings, peak-estimate caveat, no-nsys; claims audit already scans README+WRITEUP percentages.
- Metric: cold clone (fresh git clone, README NMake+VsDevCmd commands):
  configure OK, full CUDA build OK, ctest 3/3 pass, headline v6-1024^3
  0.308224 ms vs committed 0.305664 ms (ratio 1.008, bar was 10%).
  Tables fresh (72 rows), audit passes (736 traced), plot covers 9 shapes.
- Next: commit Phase 5, push, final report.

## 2026-10-09 — Strict-IEEE default: reruns Q/U/V, reported run V, fast-math control
- Bottleneck: committed numbers were measured with `--use_fast_math` (retained
  N/O JSONs record it in `compile_flags`, build at `a2e6c0a`) — the repo rule
  is strict IEEE by default (README carve-out covered the old numbers, but the
  default itself was unmeasured).
- Change: `KERN_FAST_MATH` CMake option, default OFF (commit `6d9dbb0`; only
  delta vs `a2e6c0a` besides docs is the verified-neutral clang-format pass —
  no semantic kernel change); rebuilt strict; `kern_bench --all` runs Q
  (4 shapes), U/V (9 shapes, seed 0), each copied aside before the next run;
  new `scripts/compare_runs.py` (fast-math N/O vs strict V, 72 cells).
- Metric: U-vs-V drift flagged 8 cells above 5% (worst 32.2% K512-v1: U burst
  1.7105 vs N/O/V ~1.16). V matches the multi-run majority on 7 of 8; the 8th
  (rect 512x1024x1024 v0, drift 21.9%) is bimodal N/U ~1.64 vs O/V ~1.28 —
  both modes disclosed in WRITEUP S13; reported run V takes the O-side mode.
  Control: 8/72 cells above 5%, ALL on strict-unstable cells (largest two are
  the N-side burst modes); v4/v5/v6 within noise on all 9 shapes. Flag effect
  indistinguishable from WDDM session noise — strict stays default.
  Reported run V 1024^3: v0 2.6367 / v1 2.2912 / v2 1.1295 / v3 1.3083 /
  v4 0.6397 / v5 0.4248 / v6 0.3097 (6934 GFLOPS, 69% of cuBLAS) /
  cuBLAS 0.2128. Tables (72 rows) + plot regenerated; audit passes (736).
- ncu correction, owned: "even --query-metrics fails" was wrong — listing
  works (1MB inventory committed as `results/ncu/query_metrics_2025.3.1.txt`;
  confirms `run_ncu.sh` sample names/sections for ncu 2025.3.1); only counter
  *collection* is blocked (`ERR_NVGPUCTRPERM`). WRITEUP S13 fixed. Duplicate
  `list_sections_2025.3.1.txt` (byte-identical to committed `.log`) deleted.
  `refresh_vall.py` (temp) NOT run as-is: 20/21 anchors already applied by
  earlier sibling scripts — only its NEGATIVE_RESULTS line was still pending
  (applied manually after verifying 6/9, worst +15.8%).
- Residual gap, stated plainly: correctness + sanitizer gates were last run on
  the fast-math binary; strict-binary re-run NOT RUN (GPU execution stopped
  by policy (no admin or device changes) after the `nvidia-smi` permission failure).
  When unblocked: `build/kern_correctness --seed 1234` + sanitizer script,
  then amend this entry.
- Next: commit (evidence + prose), push, final report.

## 2026-10-15 — Correctness + sanitizer on the strict-IEEE build
- Bottleneck: the correctness and sanitizer logs came from the fast-math
  binary, so "clean" was unproven for the default strict-IEEE build that
  produced every committed bench number.
- Change: `run_sanitizer.sh` now writes a provenance header into each log
  (commit `13f4a39`, flag-extraction fix `c2d0a13`). Clean rebuild
  (`--clean-first`, NMake, Release, `KERN_FAST_MATH=OFF`; nvcc command line
  has `-O3 -lineinfo -Xcompiler=/O2`, sm_120, no `--use_fast_math`). Kernel,
  test, and CMake sources are unchanged since `4fffa19`; only the script
  changed between the build commit and the logged commit.
- Metric: `kern_correctness --seed 1234` ALL PASS (bad=0, 160 checks);
  memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors, initcheck 0
  errors, 160 PASS lines in every log; total sanitizer wall time 3m33s.
  Binary SHA-256 `e070d058…d390` in every header. `nvidia-smi` read-only
  query worked (driver 617.14). Docs now name the strict build; stale
  "126 checks / 6 kernels" counts in WRITEUP fixed to 160.

## 2026-10-15 — ncu counters work: ladder profiled at 1024^3, two explanations corrected
- Bottleneck: every mechanism claim (occupancy for v4, bank conflicts for
  v6, "tensor cores" for the cuBLAS gap) was inference, because counters
  were blocked. Today `ncu --query-metrics` and a test collection on
  `v6_kernel` both worked (no `ERR_NVGPUCTRPERM`); I did not change any
  permission or clock setting.
- Change: `scripts/run_ncu_ladder.py` (v2/v4/v5/v6/cuBLAS inside kern_bench,
  first timed launch, metric names checked against `--query-metrics`,
  LaunchStats/Occupancy sections checked against `--list-sections`,
  `--clock-control none`, local paths redacted) + `scripts/ncu_summary.py`
  (generated table). Ran twice; kept both runs. Claims audit now also
  traces numbers against the ncu summaries.
- Metric (summary.md, 1024^3): v2 and v4 both 66.67% theoretical occupancy,
  so v4's 1.8x is NOT an occupancy win; shared-load instructions fell
  41,943,040 -> 12,582,912 and FMA pipe went 8.65% -> 20.92% active. v2
  shared-load bank conflicts = 0 (claim confirmed). v5 and v6 have identical
  shared-load conflicts (4,194,304), so the transposed-A "broadcast" story
  is NOT supported; v6's measured change is in the fill (store conflicts
  1,588,225 -> 278,453) and fewer instructions. cuBLAS picks
  `cutlass_80_simt_sgemm_128x64_8x5_nn_align1` (SIMT, not tensor cores),
  same FFMA count as v6, FMA pipe 60.42% vs 33.91%.
- Repeatability: instruction and shared-memory counts reproduce (exact or
  ~1%); DRAM bytes do not (v6 9.3 vs 12.9 MB read), so DRAM is unused.
- Docs corrected (README key lessons + limitations, WRITEUP §5/7/8/9/10/13,
  NEGATIVE_RESULTS #9/#10, ENVIRONMENT). Not done: a
  float4-fill untransposed v6 variant to isolate the transpose (NOT RUN).
