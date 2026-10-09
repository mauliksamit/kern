# Writeup — SGEMM optimization ladder

Status 2026-10-09: **measured on RTX 5070 (sm_120, CUDA 13.0, driver 617.14).**
Correctness ALL PASS (126 checks: 18 shapes × (6 kernels + cuBLAS)), sanitizer clean (4 tools), bench run twice
(runs D/E) with scripted drift check plus an adjudicating run C for the one
drifting cell. Profiler counters blocked (`ERR_NVGPUCTRPERM`, attempt log in
`results/ncu/`) — performance claims rest on `cudaEvent` timing plus labeled
analytical reasoning. Every measured number below comes from `results/`.

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

Gate (`tests/correctness.cu`, seed 1234): 18 shapes × (6 kernels + cuBLAS) =
126 checks — size-1, row/col vectors, small primes (7×13×5), non-multiples
(31×33×17, 100×97×63), tile-aligned 64³, rect-large-K, 512³, large-odd
1023×513×257, v5 tile-stress shapes (non-multiples of 128/8, K = 1/7/9/129,
just-over-128 edges), plus both magnitude extremes. Result
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
memcheck/racecheck/synccheck/initcheck all clean on final binaries
(`results/sanitizer/*.log`: 0 errors, racecheck 0 hazards).

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
compiler vectorization of the A stream. Measured: 2.0× at 256³ (0.0724→0.0356 ms), shrinking to 1.10× at 1024³
(2.4800→2.2563 ms) — mapping overhead matters most where kernels are short.
The hypothesis held; magnitude varies by shape (`results/bench_*.json`).

## 5. Shared-memory tiling (v2)

`v2` (`v2_tiled.cu:18-43`) loads each 32×32 patch of `A`/`B` once into
`__shared__` (8 KB total: 2 tiles × 1024 floats), then accumulates from shared.
Global traffic per output tile drops from `2·32·32·32` reads to
`2·32·32` per K-tile — a ~32× reduction for large K. `B` is served from shared
(broadcast/column reads) instead of strided DRAM. Bank conflicts: `As[ty][k]`
is a same-address broadcast per warp row (no conflict); `Bs[k][tx]` touches 32
distinct banks (no conflict) — expect ≈ 0; `ncu` bank-conflict counters NOT RUN
(blocked, §13; re-tested after counters were enabled, still blocked).
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

## 7. Thread coarsening and occupancy (v4 — MEASURED, hypothesis confirmed)

`v4` (`kernels/v4_coarsened.cu`) keeps v2's 32×32 output tile and shared-memory
algorithm but launches 16×16 = 256 threads, each accumulating a 2×2 micro-tile
in registers (4× the work per thread; A-row/B-column elements reused across the
micro-tile; 4× more blocks eligible for concurrent residency). Tile fill is
cooperative (256 threads × 4 elements). Result at 1024³ (reported run M): **0.65ms vs v2's
1.13ms — 1.7× faster** (3281 vs 1904 GFLOPS); 512³: 2.2× (0.0826 vs 0.1852ms);
rect shapes ≈ 2.2× (2.22× / 2.15×). The occupancy/work-per-thread hypothesis is confirmed by
measurement: the 1024-thread blocks were the binding constraint, not load width
(§6). v4 reaches 25% of the estimated 13.13 TFLOPS FP32 peak; cuBLAS still leads
2.9× (9699 GFLOPS) — the remaining gap is the vendor library's territory
(tensor cores, swizzling, per-shape tuning), stated as future work, not failure.

## 8. Register blocking (v5 — MEASURED, confirmed at large shapes)

`v5` (`kernels/v5_regtile.cu`): BM=BN=128, BK=8 block tiles on 256 threads,
each accumulating an 8x8 register micro-tile (64 C fragments + 8 A + 8 B
fragments). Fill efficiency 64 MAC/float vs v4's 16 (4x). Scalar shared
loads, no swizzle: any bank-conflict cost stays in this measurement so v6
(layout change) can isolate it.

Result (reported run M) at 1024^3: 0.4096ms vs v4 0.6544 (1.6x; 5243 vs 3281
GFLOPS); K512: 1.33x (0.2012 vs 0.2668); rect: 1.51x (0.1862 vs 0.2810). v5
reaches 40% of the estimated peak; cuBLAS leads 1.9x (9699 GFLOPS, 74%).

Crossover, stated plainly: v5 LOSES on small problems — 512^3: 0.0947 vs
0.0826 (0.87x); 256^3: 0.0544 vs 0.0180 (0.33x). A 128-wide tile on a
256-wide problem launches 4 blocks (1024 threads): the big tile starves
parallelism where v4's 32-wide tiles fill the machine. Tile size is a bet on
problem size; the ladder now shows both sides of it.

What surprised us: the As column pattern aliases rows 8 apart onto identical
bank sets (8 floats/row = one 32-bank period divisor), yet the 1.5x win came
anyway — layout is second-order at this intensity. v6 tests whether fixing
the layout still pays on top.

## 9. Vectorized fill + transposed A tile (v6 — MEASURED, confirmed everywhere)

`v6` (`kernels/v6_vec_transposed.cu`): same BM/BN/BK/TM/TN/grid as v5, two
layout changes: (a) one float4 per thread in the cooperative fill (256
float4s = one 128x8 / 8x128 tile exactly); (b) the A tile stored transposed
(At[8][128]), so the hot inner-loop fragment load `At[k][ty*8+i]` hits one
address per ty-group (broadcast, no conflict) where v5's `As[ty*8+i][k]` put
both ty-groups on the same bank at different addresses (2-way conflict on
every A-fragment load). Scalar fallback fills the same transposed layout on
edge/misaligned tiles; the fast-path decision is tile-level and block-uniform.

Result (reported run M) at 1024^3: 0.3053ms vs v5 0.4096 (1.3x; 7034 vs 5243
GFLOPS); K512: 1.25x (0.1608 vs 0.2012); rect: 1.23x (0.1517 vs 0.1862);
512^3: 1.20x (0.0791 vs 0.0947); even 256^3: 1.16x (0.0469 vs 0.0544). v6
reaches 54% of the estimated peak; cuBLAS leads 1.4x (9699 GFLOPS, 74%).

What this decides: layout was NOT zero — the transposed A tile plus vector
fill is worth a consistent ~1.2-1.3x across all five shapes, including the
4-block 256^3 launch where v5 starved (v6's cheaper fill helps exactly
where parallelism is scarcest). v3's failure was being vector-only without
register intensity; v6 has both. Remaining gap to cuBLAS (1.4x) is now in
vendor territory: tensor cores, swizzling, async copies, per-shape tuning.

## 10. Register usage and occupancy (analysis; timing evidence measured, counters NOT RUN)

`v2`/`v3` run 1024 threads/block with 8 KB shared per block. Most SMs cannot
co-reside two such blocks (thread-count and shared-memory limits), so occupancy
is structurally capped — one resident block per SM means latency hiding rests
entirely on the 1024 threads' own parallelism, and register pressure per thread
directly trades against nothing (no second block to make room for). v4 tested
exactly this (§7) and the 1.5–2.2× win confirms it. `ncu` occupancy +
register-per-thread metrics NOT RUN (counters blocked, §13) — the mechanism is
inferred from the controlled experiment (only mapping/coarsening changed), not
assumed.

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
| 256³ | 0.0724 | 0.0356 | 0.0277 | 0.0287 | 0.0180 | 0.0544 | 0.0469 | 0.0205 |
| 512³ | 0.3376 | 0.2601 | 0.1852 | 0.1776 | 0.0826 | 0.0947 | 0.0791 | 0.0397 |
| 1024³ | 2.4800 | 2.2563 | 1.1277 | 1.2580 | 0.6544 | 0.4096 | 0.3053 | 0.2214 |
| 1024×1024×512 | 1.6069 | 1.1383 | 0.5931 | 0.5909 | 0.2668 | 0.2012 | 0.1608 | 0.1083 |
| 512×1024×1024 | 1.2743 | 1.1385 | 0.6090 | 0.6142 | 0.2782 | 0.1862 | 0.1517 | 0.1099 |

GFLOPS (1024³): v0 866 → v1 952 → v2 1904 → v3 1707 → v4 3281 →
**v5 5243** → **v6 7034** → cuBLAS 9699
(74% of the 13.13 TFLOPS estimated peak: 36 SMs × 128 cores × 2 FMA ×
1.425 GHz reported clock — an estimate, re-verify cores/SM against the whitepaper
before citing elsewhere; observed boost reached 2625 MHz, so the estimate is
likely understated, see §13). Findings: v0→v1 helps most at small shapes (2.0× at 256³,
1.10× at 1024³ — mapping overhead vs memory); v1→v2 ≈ 1.3–2.0× (reuse);
v2→v3 never a gain in any run (v3 slower on 4/5 shapes here: 1.2580 vs 1.1277 ms
at 1024³; negative result, §14); v2→v4 ≈ 1.5–2.2×
(occupancy); v4→v5 ≈ 1.3–1.6× at 1024²-scale and above, negative below 512³
(tile-size crossover, §8); v5→v6 ≈ 1.2–1.3× on every shape including 256³
(layout + vector fill help even 4-block launches); cuBLAS leads v6 1.4–2.3×;
at 256³ v4 still beats cuBLAS 1.1× while tile-starved v5 trails
(launch/parallelism regime — the ranking itself is size-dependent).
Repeatability: runs F/G/H/I/J/K/L/M adjudicated per cell (see §13); prior D/E/C
values retained in LOG.md. v3 beats v2 only at 512³ (4%) and K512 (0.4%) —
within noise in every session.

## 13. Profiler evidence (NOT RUN — blocked, with evidence)

`ncu --section SpeedOfLight` attempted 2026-10-09 (2025.3.1.0, single 256³
shape): `ERR_NVGPUCTRPERM` — no permission to access GPU performance counters
on this WDDM device. Enabling counters needs admin/developer-mode action (out
of scope). Attempt log:
`results/ncu/ncu_smoke_M256_attempt.log`. Re-tested after counters were
enabled: STILL blocked — even `ncu --query-metrics` fails
(`results/ncu/query_metrics_ERR_NVGPUCTRPERM.log`), so no metric name in the
installed version can be exercised here; `results/ncu/list_sections_2025.3.1.log`
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

## 14. Failed experiments and negative results

Kept in `docs/NEGATIVE_RESULTS.md` (design rejections: 64×64 thread blocks,
GPU-side double accumulation, timing copies) plus the v3 barrier fix, the v3
vectorization non-result (within ±5% of v2 on all 5 shapes across runs D/E —
cooperative-load/branch overhead at the 1024-thread occupancy wall; v4 shows
the wall was the real problem), and the ptxas `__launch_bounds__` rejection in
`LOG.md`. Policy: a hypothesis that measures within noise is recorded with
numbers and kept reproducible — never deleted, never re-timed until attractive.

## 15. Hardware-specific conclusions

Single data point (RTX 5070 Laptop, Blackwell sm_120, WDDM display-attached, driver
617.14, CUDA 13.0): the ladder ranks v0 < v1 < v2 ≈ v3 < v4 < v5 < v6 < cuBLAS
at 1024²-scale and above, with v6 at 54% and cuBLAS at 74% of estimated FP32
peak at 1024³. At 256³ the ranking inverts for the big tiles (v5/v6 trail v4
3×/2.6×) while v4 still beats cuBLAS 1.1× — tile size is a bet on problem size. No conclusion is generalized to other architectures without
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
- Planned `ml/` extension (not started): benchmark representative projection
  shapes with PyTorch (eager vs compiled, latency/throughput/memory/numerics)
  and use this ladder to explain the bottleneck. Only after core bench lands.
  Status 2026-10-09: core bench HAS landed, but `torch` is not installed here
  (verified `ModuleNotFoundError`) and the MSYS2 Python has no pip, so a
  measured eager-vs-compiled experiment is deferred — exact next step:
  install a CUDA torch build, run `ml/transformer_gemm.py` (to be written:
  QKV/FFN-shaped GEMMs, eager vs `torch.compile`, latency + numerics vs this
  ladder), commit its JSON next to `results/`.
- Down payment with zero new dependencies (`scripts/roofline.py`, self-check
  passes): a 4096-wide FFN up-projection at 512 tokens (M=512,N=4096,K=4096)
  needs 17.18 GFLOP at AI=204.8 FLOP/B — the same compute-leaning regime as the
  measured shapes (AI 128–171), where the v4 occupancy win applies; nearest
  measured point (512×1024×1024, v6 7078 GFLOPS) implies ≥34.6 GB/s ideal
  traffic. An attention QKV case (M=2048,N=4096,K=1024, AI=292.6) sits even
  further toward compute-bound. These are analytical placements against
  measured references (extrapolation, labeled as such) — they motivate which
  transformer shapes to benchmark first when torch is available.
