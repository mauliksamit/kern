# Writeup — SGEMM optimization ladder

Status 2026-10-09: **measured on RTX 5070 (sm_120, CUDA 13.0, driver 617.14).**
Correctness ALL PASS (160 checks: 20 shapes × (7 kernels + cuBLAS)) and sanitizer clean (4 tools), both on the
default strict-IEEE build (re-run 2026-10-15; provenance in each `results/sanitizer/*.log` header). Bench run twice
(runs D/E) with scripted drift check plus an adjudicating run C for the one
drifting cell. Profiler counters were blocked until 2026-10-15; since then a
per-kernel Nsight Compute profile at 1024³ exists (§13, `results/ncu/ladder_M1024_N1024_K1024/summary.md`) and it
corrected two of our earlier explanations (§7, §9). Every measured number
below comes from `results/`.

## 1. Problem definition and row-major representation

Compute `C = A·B` in FP32 with all matrices row-major: `A` is M×K, `B` is K×N,
`C` is M×N, element `C[i][j] = Σ_k A[i*K+k]·B[k*N+j]`. Row-major means row `i`
of `A` occupies contiguous addresses `A[i*K .. i*K+K)`; column `j` of `B`
(`B[k*N+j]` over `k`) is strided by `N`. That asymmetry drives the whole ladder:
`A` is naturally streamable along rows, `B` is not, so every version is really an
answer to "how do we feed `B` to 1000s of threads without re-reading DRAM?"

## 2. Correctness methodology and numerical tolerances (MEASURED)

Ground truth: `src/matmul_cpu.cpp` accumulates each dot product in `double`,
rounding once to float. GPU kernels accumulate in `float` (deliberately — the
benchmark must reflect real SGEMM throughput, not mixed precision; rejected
alternative in `docs/NEGATIVE_RESULTS.md`).

Tolerance (`include/kern/matmul_cpu.h:near`): `|got−ref| ≤ 1e-4 + 1e-4·|ref|`.
Justification: float dot-product error grows ~`K·eps` (`eps ≈ 1.2e-7`); for
`K ≤ 1024` and `|elements| ≤ 1` the absolute bound ≈ 1e-4. The relative term
covers large magnitudes, the absolute floor covers small ones — each with a
dedicated shape (1e9–1e10 and 1e-11–1e-10 ranges).

Gate (`tests/correctness.cu`, seed 1234): 20 shapes × (7 kernels + cuBLAS) =
160 checks — size-1, row/col vectors, small primes (7×13×5), non-multiples
(31×33×17, 100×97×63), tile-aligned 64³, rect-large-K, 512³, large-odd
1023×513×257, v5 tile-stress shapes (non-multiples of 128/8, K = 1/7/9/129,
just-over-128 edges), v6 vector-path shapes (128³ full-fast, 200³ mixed),
plus both magnitude extremes. Result
(`results/correctness_seed1234.json`, `bad=0`):

| Shape class | Worst abs err (all 5 kernels agree) |
|---|---|
| 1×1×1 | 0 |
| K ≤ 17 shapes | ≤ 7.15e-07 |
| 64³ / 100×97×63 | 2.38e-06 / 3.81e-06 |
| K=512 shapes | 2.67e-05 – 3.24e-05 |
| 1023×513×257 (K=257) | 1.72e-05 |
| large-val (rel-tol regime) | 8.44e+14 abs (≈1e-5 relative — passes by design) |
| small-val (abs-floor regime) | 6.46e-26 |

Error scales with K as the model predicts; no kernel differs from any other
beyond float rounding. cuBLAS-vs-reference passes on all 12 shapes with
*smaller* worst errors than the custom kernels (e.g. 1.14e-05 vs 3.24e-05 at
512³) — the vendor kernel's blocking/accumulation order rounds better, an
independent confirmation the baseline computes the same math. Sanitizer:
memcheck/racecheck/synccheck/initcheck all clean on the final strict-IEEE
binary (`KERN_FAST_MATH=OFF`; `results/sanitizer/*.log`: 0 errors, racecheck
0 hazards, 160 PASS lines each; header records commit, flags, binary SHA-256).

## 3. Kernel execution model

All kernels map the M×N output domain onto a CUDA grid; v0–v3 give each thread one `C`
element, while v4 (§7) coarsens to a 2×2 micro-tile per thread. `v0` uses 2D 16×16 blocks over a 2D grid
(`kernels/v0_naive.cu`); `v1` flattens to a 1D grid of 256-thread blocks with
`row = tid / N, col = tid % N`; `v2`/`v3` use 2D 32×32 blocks = 1024 threads
(the architectural maximum — portable but occupancy-limited, §10). Every launch
is followed by `CUDA_CHECK_LAST_ERROR()` (`include/kern/cuda_check.h`), and
`CUDA_SYNC_DEBUG()` synchronizes in non-`NDEBUG` builds. All versions
bounds-check, so arbitrary M/N/K are correct (edge tiles zero-fill in
v2/v3, guarded stores everywhere).

## 4. Global-memory access patterns (v0 → v1)

`v0`'s inner loop (`v0_naive.cu:21-23`) reads `A[row*K+k]` (sequential per
thread, but `row`-strided across a warp — 32 different cache lines per step)
and `B[k*N+col]` (consecutive across a warp for fixed `k` — coalesced — but
re-read from DRAM for every output row, zero reuse). Arithmetic intensity ≈
2 flops / 8 bytes ≈ 0.25 FLOP/B: hopelessly DRAM-bound on any modern GPU. The
naive "B is uncoalesced" telling is imprecise — the warp's B accesses for fixed
`k` are contiguous; the actual disease is no reuse (each `B` element read M
times, each `A` element N times).

`v1` (`v1_coalesced.cu`) changes only the mapping: flattened 1D grid,
`__restrict__`, stores to contiguous `C[tid]`. A-row reads stay per-thread
private; the win, if any, comes from fully coalesced C stores and friendlier
compiler vectorization of the A stream. Measured: 2.1× at 256³ (0.0725→0.0344 ms), shrinking to 1.15× at 1024³
(2.6367→2.2912 ms) — mapping overhead matters most where kernels are short.
The hypothesis held; magnitude varies by shape (`results/bench_*.json`).
Profiler evidence for this step: NOT RUN (v0/v1 are not in the 1024³ ncu
profile, §13).

## 5. Shared-memory tiling (v2)

`v2` (`v2_tiled.cu:18-43`) loads each 32×32 patch of `A`/`B` once into
`__shared__` (8 KB total: 2 tiles × 1024 floats), then accumulates from shared.
Global traffic per output tile drops from `2·32·32·32` reads to
`2·32·32` per K-tile — a ~32× reduction for large K. `B` is served from shared
(broadcast/column reads) instead of strided DRAM. Bank conflicts: `As[ty][k]`
is a same-address broadcast per warp row (no conflict); `Bs[k][tx]` touches 32
distinct banks (no conflict) — expect ≈ 0. **Confirmed by ncu (2026-10-15,
1024³, §13):** shared-load bank conflicts = 0 on 41,943,040 shared-load
instructions. The fill *stores* do conflict (642,576 conflicts on 2,097,152
store instructions), a minor cost next to the loads. Source: `results/ncu/ladder_M1024_N1024_K1024/summary.md`.
Two `__syncthreads()` per K-tile (after fill, after compute) are required and
uniform. Non-multiples handled by predicated zero-fill loads + guarded store.

## 6. Vectorization and alignment (v3)

`v3` (`v3_tiled_vectorized.cu`) keeps the v2 structure and vectorizes only the
tile *fill*: 256 threads × `float4` = 1024 floats per tile, halving global
transactions on the fast path, with `#pragma unroll` on the compute loop and
`__launch_bounds__(1024,1)` (the `(1024,2)` hint was rejected by ptxas — SM
residency — see `LOG.md` 2026-10-03).

Correctness subtlety (the project's best bug story): the fast-path decision
must be **block-uniform** because both branches contain `__syncthreads()`.
The original code tested alignment from per-thread `(row, col)` pointers, so
threads in one block disagreed whenever `K % 4 ≠ 0` or `N % 4 ≠ 0` — divergent
barriers, undefined behavior, silent hang (binary produced zero output in
120 s). The fix computes alignment from tile-base pointers and additionally
requires `K % 4 == 0 && N % 4 == 0`, which is exactly the condition making
every row share the base pointer's 16B alignment (float = 4B; element index
`p` is 16B-aligned iff `p % 4 == 0`). Post-fix: full gate ALL PASS in <45 s.
Reference: CUDA Programming Guide barrier-uniformity rule
(`docs/REFERENCES.md` #7). Edge/misaligned shapes take the scalar fallback and
are covered by the awkward-shape gate.

## 7. Thread coarsening (v4 — MEASURED; speedup confirmed, occupancy explanation refuted by ncu)

`v4` (`kernels/v4_coarsened.cu`) keeps v2's 32×32 output tile and shared-memory
algorithm but launches 16×16 = 256 threads, each accumulating a 2×2 micro-tile
in registers (4× the work per thread; A-row/B-column elements reused across the
micro-tile; 4× more blocks eligible for concurrent residency). Tile fill is
cooperative (256 threads × 4 elements). Result at 1024³ (reported run V): **0.64ms vs v2's
1.13ms — 1.8× faster** (3357 vs 1901 GFLOPS); 512³: 2.2× (0.0817 vs 0.1813ms);
rect shapes ≈ 2.2× (2.18× / 2.24×). **The speedup is real; the original
explanation (occupancy) was wrong.** The pre-counter write-up credited
"occupancy + work per thread"; ncu at 1024³ (§13, `results/ncu/ladder_M1024_N1024_K1024/summary.md`) shows
theoretical occupancy is identical — 66.67% for both (v2: 1 block × 32 warps;
v4: 4 blocks × 8 warps, register-limited) — and achieved occupancy is
66.32% vs 63.61%. What changed is work per instruction: shared-load
instructions fell from 41,943,040 to 12,582,912 (70% fewer), total warp
instructions from 99,942,400 to 74,686,464, and the FMA pipe went from
8.65% to 20.92% active, for the same 1,073,741,824 thread FFMAs. So the 2×2
register micro-tile wins by reusing each shared-memory value twice, not by
fitting more warps. v4 reaches 33% of cuBLAS; cuBLAS still leads
3.0× (10093 GFLOPS); see §9 for what ncu shows about that gap.
Cost side, also from ncu: v4's cooperative fill has 6,522,829 shared-store
bank conflicts (v2: 642,576), a remaining inefficiency.

## 8. Register blocking (v5 — MEASURED, confirmed at large shapes)

`v5` (`kernels/v5_regtile.cu`): BM=BN=128, BK=8 block tiles on 256 threads,
each accumulating an 8x8 register micro-tile (64 C fragments + 8 A + 8 B
fragments). Fill efficiency 64 MAC/float vs v4's 16 (4x). Scalar shared
loads, no swizzle: any bank-conflict cost stays in this measurement so v6
(layout change) can isolate it.

Result (reported run V) at 1024^3: 0.4248ms vs v4 0.6397 (1.5x; 5055 vs 3357
GFLOPS); K512: 1.28x (0.2109 vs 0.2708); rect: 1.49x (0.1885 vs 0.2804). v5
reaches 50% of cuBLAS; cuBLAS leads 2.0x (10093 GFLOPS).
ncu at 1024³ (§13): 168 registers/thread limit v5 to 1 block (8 warps) per
SM, theoretical occupancy 16.67% (achieved 16.55%), and the 64-block grid is
1.78 waves on 36 SMs. In exchange, shared-load instructions drop to
2,097,152 (from v4's 12,582,912) and the FMA pipe rises to 28.62% active.

Crossover, stated plainly: v5 LOSES on small problems — 512^3: 0.0946 vs
0.0817 (0.86x); 256^3: 0.0573 vs 0.0176 (0.31x). A 128-wide tile on a
256-wide problem launches 4 blocks (1024 threads): the big tile starves
parallelism where v4's 32-wide tiles fill the machine. Tile size is a bet on
problem size; the ladder now shows both sides of it.

Pre-counter analysis (kept for the record, partly refuted in §9): we predicted
the As column pattern aliases rows 8 apart onto identical bank sets and costs
a 2-way conflict on A-fragment loads. ncu does show 4,194,304 shared-load bank
conflicts in v5 — but v6, which was designed to remove them, shows exactly
the same count (§9).

## 9. Vectorized fill + transposed A tile (v6 — MEASURED speedup; bank-conflict explanation refuted by ncu)

`v6` (`kernels/v6_vec_transposed.cu`): same BM/BN/BK/TM/TN/grid as v5, two
layout changes: (a) one float4 per thread in the cooperative fill (256
float4s = one 128x8 / 8x128 tile exactly); (b) the A tile stored transposed
(At[8][128]), so the hot inner-loop fragment load `At[k][ty*8+i]` hits one
address per ty-group (broadcast, no conflict) where v5's `As[ty*8+i][k]` put
both ty-groups on the same bank at different addresses (2-way conflict on
every A-fragment load). Scalar fallback fills the same transposed layout on
edge/misaligned tiles; the fast-path decision is tile-level and block-uniform.

Result (reported run V) at 1024^3: 0.3097ms vs v5 0.4248 (1.4x; 6934 vs 5055
GFLOPS); K512: 1.25x (0.1685 vs 0.2109); rect: 1.27x (0.1487 vs 0.1885);
512^3: 1.19x (0.0793 vs 0.0946); even 256^3: 1.21x (0.0472 vs 0.0573). v6
reaches 69% of cuBLAS; cuBLAS leads 1.5x (10093 GFLOPS).

What the timing decides: v6 is a consistent ~1.1-1.4x faster than v5 on
all nine shapes, including the 4-block 256^3 launch where v5 starved.

What ncu changes (1024³, §13, `results/ncu/ladder_M1024_N1024_K1024/summary.md`): the predicted mechanism — the
transposed A tile turning 2-way-conflicted fragment loads into broadcasts —
is **not supported**. v5 and v6 have identical shared-load counts:
2,097,152 instructions, 10,485,760 wavefronts, 4,194,304 bank conflicts,
reproduced exactly in two profiling runs (`..._run1/`). The measured
differences are all on the fill side and in instruction count: shared-store
instructions 524,288 → 327,680, store wavefronts 2,112,513 → 802,741, store
bank conflicts 1,588,225 → 278,453, total warp instructions 42,288,640 →
38,950,912, registers 168 → 138, FMA pipe 28.62% → 33.91% active. So the
measured gain comes from the cheaper vectorized fill; whether the transpose
itself contributes anything is not isolated (that needs a float4-fill,
untransposed variant: NOT RUN). Recorded in `docs/NEGATIVE_RESULTS.md`.

The cuBLAS gap (1.4x) is also not what we guessed. The kernel cuBLAS picks
here is `cutlass_80_simt_sgemm_128x64_8x5_nn_align1`, a SIMT (CUDA-core)
kernel, not tensor cores, and it executes the same work: 1,075,838,976
thread FFMAs vs v6's 1,073,741,824 and a similar warp-instruction count
(42,681,600 vs 38,950,912). The difference is how busy the FMA pipe is:
60.42% active vs 33.91%. Measured contributors: 25% vs 16.67% theoretical
occupancy (128 vs 138 registers, 128-thread blocks), 384 blocks = 3.56 waves
vs 64 blocks = 1.78 waves (less tail imbalance), and 4,320,812 shared-load
wavefronts vs 10,485,760. The grid's z-dimension of 3 suggests split-K and
"8x5" in the CUTLASS name suggests a 5-stage pipeline; both are read from the
name and launch shape, not measured.

## 10. Register usage and occupancy (MEASURED with ncu, 1024³)

Pre-counter claim: v2's 1024-thread blocks cap residency at one block per SM,
so occupancy limits v2 and v4 wins by fixing it. ncu (§13): the first half is
true (v2's register, shared-memory and warp limits are each 1 block/SM), but
one 32-warp block already gives 66.67% theoretical occupancy (sm_120 allows
48 warps/SM), the same as v4's 4 × 8 warps. Occupancy did not change between
v2 and v4; instruction count did (§7). Occupancy only drops at v5/v6
(16.67%, register-limited at 168/138 registers) and that is the price of the
register tile. The cuBLAS kernel sits between (25%). Registers per thread:
v2 38, v4 56, v5 168, v6 138, cuBLAS 128.

## 11. Benchmark methodology and its limitations

`src/bench.cu`: `cudaEvent` around the kernel only (H2D/D2H outside the timed
region — copies are PCIe bandwidth, not kernel performance); ≥10 warmup, ≥100
timed iterations (10 if a single run exceeds 100 ms, flagged in JSON and
stdout); median/min/p95/mean over identical inputs across versions; derived
metrics defined in code — `GFLOPS = 2·M·N·K / t`, `GB/s = (MK+KN+MN)·4 / t`
(ideal traffic, no reuse — an upper-bound model, not a claim); estimated FP32
peak from device properties (`SMs × 128 cores × 2 FMA × clock`, labeled an
estimate — cores/SM must be re-verified per architecture whitepaper); raw
per-iteration arrays + metadata (git commit, device, CC, flags pointer, sizes,
seed) in `results/bench_*.json`; protocol requires every bench run twice with
>5% median-drift investigation. cuBLAS baseline (`time_cublas`) uses the
standard dual-transpose emulation (`C^T = B^T·A^T`) — same math, column-major
API. Limitations: display-attached WDDM GPU (scheduling noise — hence medians
and repeats); no `nsys` (kernel time only, no end-to-end traces); `ncu`
counters may be restricted under WDDM (then profiler evidence stays NOT RUN); shapes stop at 1024.

## 12. Measured performance comparisons

Bench matrix: 256/512/1024 square + 1024×1024×512 + 512×1024×1024,
`kern_bench --all --seed 0`. History: runs D/E (+run C) established the baseline;
the current session ran F/G/H/I under heavier display-GPU contention (see §13).
Reported run: G (`results/bench_*.json`); table via
`scripts/generate_tables.py --write` into `README.md`; audit
(`scripts/claims_audit.py`) passes. Medians (ms):

| Shape | v0 | v1 | v2 | v3 | v4 | v5 | v6 | cuBLAS |
|---|---|---|---|---|---|---|---|---|
| 256³ | 0.0725 | 0.0344 | 0.0281 | 0.0282 | 0.0176 | 0.0573 | 0.0472 | 0.0188 |
| 512³ | 0.3394 | 0.2621 | 0.1813 | 0.1784 | 0.0817 | 0.0946 | 0.0793 | 0.0395 |
| 1024³ | 2.6367 | 2.2912 | 1.1295 | 1.3083 | 0.6397 | 0.4248 | 0.3097 | 0.2128 |
| 1024×1024×512 | 1.6134 | 1.1591 | 0.5999 | 0.5962 | 0.2756 | 0.2109 | 0.1685 | 0.1112 |
| 512×1024×1024 | 1.2835 | 1.1547 | 0.6266 | 0.6154 | 0.2804 | 0.1885 | 0.1487 | 0.1125 |
| 2048x2048x2048 | 23.8602 | 22.2106 | 10.4853 | 11.0761 | 4.7135 | 2.7826 | 2.2613 | 1.5808 |
| 4096x4096x4096 | 408.7477 | 314.9752 | 96.4199 | 98.6119 | 45.8335 | 25.2083 | 22.6764 | 13.5843 |
| 4096x512x1024 | 5.2595 | 5.8387 | 2.2587 | 2.4696 | 1.3018 | 0.7723 | 0.5849 | 0.4115 |
| 512x4096x1024 | 5.1474 | 5.6516 | 2.2481 | 2.4204 | 1.2828 | 0.7691 | 0.5955 | 0.4626 |

GFLOPS (1024³): v0 814 → v1 937 → v2 1901 → v3 1641 → v4 3357 →
**v5 5055** → **v6 6934** → cuBLAS 10093.
(Analytical note, not a claim: 13.13 TFLOPS is an estimate from the
driver-reported 1425 MHz SM clock (36 SMs × 128 FP32 cores × 2 FMA — cores/SM
re-verify per whitepaper). Read-only clock samples in the committed JSONs show
real boost far above that: max observed 2730 MHz (tall-skinny run), for an
implied 25.16 TFLOPS by the same formula, and 2505 MHz on the 1024³ run (23.09
TFLOPS implied). Headline comparisons therefore use percent of cuBLAS, a
measured baseline, instead of percent of this estimate. Findings: v0→v1 helps most at small shapes (2.1× at 256³,
1.15× at 1024³ — mapping overhead vs memory); v1→v2 ≈ 1.2–3.3× (reuse; widest at 4096³ where the 1D mapping collapses);
v2→v3 never a gain in any run (v3 slower on 6/9 shapes here: 1.31 vs 1.13 ms
at 1024³; negative result, §14); v2→v4 ≈ 1.6–2.2×
(occupancy); v4→v5 ≈ 1.3–1.8× at 1024²-scale and above, negative below 512³
(tile-size crossover, §8); v5→v6 ≈ 1.1–1.4× on every shape including 256³
(layout + vector fill help even 4-block launches); cuBLAS leads v6 1.3–2.5×;
at 256³ v4 still beats cuBLAS 1.1× while tile-starved v5 trails
(launch/parallelism regime — the ranking itself is size-dependent).
Repeatability: runs F/G/H/I/J/K/L/M/N/U/V adjudicated per cell (see §13); prior D/E/C
values retained in LOG.md. v3 beats v2 at 512³ (2%), K512 (1%) and rect (2%) —
within noise in every session.

## 13. Profiler evidence (MEASURED 2026-10-15 at 1024³; earlier attempts blocked)

**2026-10-15: counter collection works.** `ncu --query-metrics` succeeds and
a test collection on `v6_kernel` returned values (no `ERR_NVGPUCTRPERM`).
`scripts/run_ncu_ladder.py` profiled v2, v4, v5, v6 and cuBLAS inside
`kern_bench` at 1024³ (first timed launch of each, after warmup),
using only metric names present in `ncu --query-metrics` plus the
`LaunchStats`/`Occupancy` sections from `ncu --list-sections`, with
`--clock-control none` (the repo never locks clocks). `scripts/ncu_summary.py`
generates `results/ncu/ladder_M1024_N1024_K1024/summary.md` from the CSVs. The run was repeated
(`results/ncu/ladder_M1024_N1024_K1024_run1/`): instruction counts and
shared-load wavefronts/conflicts match exactly or within about 1%; DRAM byte
counts do **not** reproduce (v6 read 9,293,312 vs 12,862,720 bytes across
the two runs), so no claim in this write-up uses DRAM bytes. Durations under
ncu are not speed claims (clocks unlocked, one launch); speed comes from the
cudaEvent JSONs. Findings are in §5, §7, §8, §9, §10. The historical record
of the blocked attempts follows.


`ncu --section SpeedOfLight` attempted 2026-10-09 (2025.3.1.0, single 256³
shape): `ERR_NVGPUCTRPERM` — no permission to access GPU performance counters
on this WDDM device. Enabling counters needs admin/developer-mode action (out
of scope). Attempt log:
`results/ncu/ncu_smoke_M256_attempt.log`. Re-tested after counters were
enabled: collection STILL blocked — any counter access returns
`ERR_NVGPUCTRPERM` (`results/ncu/query_metrics_ERR_NVGPUCTRPERM.log`).
Metric *listing*, which needs no counter permission, does work:
`results/ncu/query_metrics_2025.3.1.txt` is the full installed-version
metric inventory — it confirms the `run_ncu.sh` sample names
(`sm__throughput`, `dram__throughput`) and the `SpeedOfLight` /
`MemoryWorkloadAnalysis` sections exist in ncu 2025.3.1, so the
name-confirmation rule is satisfied even though no counter value can be
collected here; `results/ncu/list_sections_2025.3.1.log`
records the section inventory only. `scripts/run_ncu.sh` was NOT run (same
root cause, no data possible). Full per-version metric plan
(DRAM transactions, coalescing efficiency, bank conflicts, occupancy) is kept
in `scripts/run_ncu.sh` for a machine with counter access. Repeatability notes
from timing distributions (substitute evidence, labeled as such): 1024³ v2
medians A/B/C = 1.1175/1.2236/1.1352 with mins 1.0956/1.1294/1.0981 — the 9.5%
A-vs-B drift sits inside overlapping min–p95 bands (WDDM display-GPU noise from
compositor/background apps sharing the GPU); run B kept as reported run, all
three values disclosed here. The single D-vs-E marginal cell (K512 v4, 5.42%,
mins identical at 0.2601/0.2613) got the same treatment.

New session (runs F/G/H/I, heavier contention): F-vs-G drift flagged 8 cells,
worst 46% (1024³ v2: F 2.0918 vs G 1.1263). The new env samples explain why
this is machine state, not kernel behavior — run F's clocks fell 1627→1447 MHz
across the run while run G's rose 1627→2377 MHz (power 21→71 W, near the 75 W
cap); long kernels (v0/v1/v2 at 1024³) caught the swings differently per run.
Newer sessions (runs J/K, then L/M, same contention): J-vs-K drift flagged 2
cells (1024³ v0 8.3% with identical mins 2.4620, v5 5.9% with K confirmed by
an independent first-look 0.4308); L-vs-M flagged 6 (all 1024³ long kernels
burst-hit in run L against history-matched run M). Reported run M has the
fewest history-disagreeing cells. F/G/H/I/J/K/L medians are disclosed here
and in LOG.md; their raw JSONs live outside the repo (temp dir) so the
generated table keeps one raw run per shape.
Process note: run E's JSONs were overwritten by run F before being copied
aside — E values survive only in prose (prior LOG/WRITEUP); all later runs
are copied aside before the next run starts.
Newest session (runs N/O/P, 9 shapes): N-vs-O drift flagged 12 cells, all on
multi-ms kernels (v0/v1 at 1024³/rect/tall, cublas at 1024³/tall/wide); every
v2–v6 cell except two marginals (2048 v2 6.0%, 1024³ v4/v5 5–6%) agreed.
Three-run adjudication + burst-free-floor analysis (per-cell iters
distributions in LOG.md): K512-v0 reads 1.25 (D/E/F, 87–94% of iters near the
1.24 floor) vs 1.61–1.63 (G/J/N/O/P, 15–44% near floor, same 1.24 floor) —
the slow mode is burst contamination shifting the median, not a machine-state
change (floors agree across all clock states). Reported run N (fewest
severely-disturbed cells); F–P medians disclosed here and in LOG.md.
Provenance note: the retained N/O JSONs record `--use_fast_math` in
`compile_flags` (pre-default-change build at `a2e6c0a`); Q/U/V record the
strict-IEEE default (build at `6d9dbb0`, no semantic kernel change —
only the verified-neutral clang-format pass between the two commits).
Strict-math session (runs U/V at 9 shapes, plus 4-shape run Q, all strict):
U-vs-V drift flagged 8 cells, worst drift 32.2% (K512 v1: U 1.7105 vs V
1.1591 — U the burst outlier; N/O history 1.1558/1.1603 sides with V).
Per-cell adjudication against N/O history: V matches the multi-run majority
on 7 of 8 (1024³ v6 V 0.3097 with N 0.3057/O 0.3008; tall-skinny cublas V
0.4115 with O 0.3966; 2048 v2/v3 inside the 9.4–11.1 strict-session spread;
256³ pair at microsecond absolute noise). The 8th cell (rect 512×1024×1024
v0, drift 21.9%) is bimodal across sessions — N/U near 1.64–1.65 vs O/V
near 1.28, both modes disclosed here; reported run V takes the O-side mode.
Fast-math control (`scripts/compare_runs.py`, fast-math N/O vs strict V,
72 cells): 8 cells above the 5% drift limit, every one coinciding with a
cell already unstable across strict-only sessions (the two largest, rect
v0/v1 at drift 22.2%/30.4%, are the N-side burst modes above); v4/v5/v6
move within noise on all 9 shapes. Verdict: the flag's effect is
indistinguishable from WDDM session noise on this workload — strict IEEE
stays the default and every committed number is now a strict-math
measurement (`compile_flags` in each JSON). Correctness and sanitizer
gates on the strict build: on 2026-10-15 a clean strict rebuild passed `kern_correctness --seed 1234`
(160/160) and all four sanitizer tools (0 errors, 0 hazards); each log
header names the commit, `KERN_FAST_MATH=OFF`, the nvcc flags and the
binary SHA-256.

## 14. Failed experiments and negative results

Kept in `docs/NEGATIVE_RESULTS.md` (design rejections: 64×64 thread blocks,
GPU-side double accumulation, timing copies) plus the v3 barrier fix, the v3
vectorization non-result (never beyond noise on any of 9 shapes across all
sessions — cooperative-load/branch overhead at the 1024-thread occupancy wall; v4 shows
the wall was the real problem), and the ptxas `__launch_bounds__` rejection in
`LOG.md`. Policy: a hypothesis that measures within noise is recorded with
numbers and kept reproducible — never deleted, never re-timed until attractive.

## 15. Hardware-specific conclusions

Single data point (RTX 5070 Laptop, Blackwell sm_120, WDDM display-attached, driver
617.14, CUDA 13.0): the ladder ranks v0 < v1 < v2 ≈ v3 < v4 < v5 < v6 < cuBLAS
at 1024²-scale and above, with v6 at 69% of cuBLAS
at 1024³. At 256³ the ranking inverts for the big tiles (v5/v6 trail v4
3.2×/2.7×) while v4 still beats cuBLAS 1.1× — tile size is a bet on problem size.
Scaling to 4096³ (64× the FLOPs of 1024³): v6 slows 73× (near-linear),
cuBLAS 64×, but v0 slows 155× (409 vs 2.64 ms) — the no-reuse kernel falls off
a cache cliff (L2-sized working sets stop fitting) while tiled kernels and
cuBLAS ride linear scaling. No conclusion is generalized to other architectures without
re-running; the `-DKERN_CUDA_ARCH` flag and `tools/detect_arch.py` exist for that.

## 16. Lessons relevant to ML inference and training

- Every transformer projection (`Q/K/V/O`, FFN up/down) and attention score
  (`Q·K^T`) is a GEMM; batch × sequence dimensions map to the M axis, model
  width to N/K. The ladder's reuse analysis (§4–5) is the same analysis that
  decides whether an inference projection is memory- or compute-bound.
- `torch.compile`/vendor kernels win the same way cuBLAS does: fusion, layout,
  and per-shape tuning — operator-level GFLOPS only matters via its effect on
  end-to-end latency, which is why this repo times kernels but documents the
  missing end-to-end trace as a limitation (§11).
- Bring-up lesson (transferable to any accelerator): silent hang → check barrier/uniformity contracts before blaming the
  toolchain; keep the wrong theory (SAC block) corrected in writing.
- Measured `ml/` extension (`ml/transformer_gemm.py`, torch 2.14.1+cu130 in
  `ml/.venv`, RTX 5070 Laptop sm_120): 5 GEMM shapes (attn projections at 512
  and 2048 tokens against d_model 4096; FFN width 11008 up/down). FP32 eager
  `torch.matmul` (TF32 explicitly disabled, CUDA events, 10+100 iters,
  median/min/p95, seed 0) reaches 9.3-10.4 TFLOPS - within 9% of the cuBLAS
  C++ harness on identical shapes (framework overhead ≈ noise); best custom
  kernel (v6) trails eager 1.5–1.8x on these shapes. Numerics vs float64
  reference: max rel err 1.2e-06–3.4e-06. A clearly labeled TF32-allowed row
  runs ~2x faster (tensor cores) and is never mixed into the FP32 comparison.
  `torch.compile` NOT RUN (`TritonMissing`, no native Windows Triton — see
  `docs/NEGATIVE_RESULTS.md` #8). Caveat: torch and C++ draws differ (own
  RNGs, same distribution) — timing is data-independent for dense GEMM;
  numerics are self-consistent within torch. Evidence:
  `results/ml_transformer.json` + `results/ml_M*_N*_K*.json`; placement via
  `scripts/roofline.py --ml`.
- Down payment with zero new dependencies (`scripts/roofline.py`, self-check
  passes): a 4096-wide FFN up-projection at 512 tokens (M=512,N=4096,K=4096)
  needs 17.18 GFLOP at AI=204.8 FLOP/B — the same compute-leaning regime as the
  measured shapes (AI 128–171), where the v4 occupancy win applies; nearest
  measured point (512×1024×1024, v6 7219 GFLOPS) implies ≥35.2 GB/s ideal
  traffic. An attention QKV case (M=2048,N=4096,K=1024, AI=292.6) sits even
  further toward compute-bound. These are analytical placements against
  measured references (extrapolation, labeled as such) — they motivate which
  transformer shapes to benchmark first when torch is available.
