# Environment (recorded 2026-10-02, scaffolding machine)

Probed by running the environment gate commands. No measurement is claimed
beyond what these commands returned.

## GPU / driver (`nvidia-smi`)

```text
NVIDIA-SMI 576.76, Driver Version: 576.76, CUDA Version: 12.9
GPU 0: NVIDIA GeForce RTX 5070 (WDDM), 8151 MiB, 59C idle
```

Update 2026-10-09: driver is now **617.14** (CUDA UMD 13.4); `nvidia-smi`
shows the RTX 5070 on WDDM, display-attached. All other toolchain rows below
re-verified today except as noted.

- GPU name: NVIDIA GeForce RTX 5070 (truncated by nvidia-smi display width; full
  name to be confirmed with `nvidia-smi --query-gpu=name --format=csv` on the bench machine)
- Inferred compute capability: **sm_120 (Blackwell GB205)** — INFERENCE, not measured
  here. RTX 50-series requires CUDA >= 12.8. Confirm after toolkit install with
  `tools/detect_arch.py` or deviceQuery, then set `-DKERN_CUDA_ARCH=sm_120` explicitly.
- Driver: 576.76. Host: AMD Ryzen AI 7 350 w/ Radeon 860M, Windows 64-bit, WDDM driver model.

## Toolchain

| Tool | Result on this machine |
|---|---|
| `nvcc --version` | **13.0.88** (installed 2026-10-03 via `winget install Nvidia.CUDA --version 13.0`) |
| `ncu --version` | **2025.3.1.0** (ships with the toolkit) |
| `nsys --version` | **NOT FOUND** (not bundled; separate Nsight Systems download) |
| `compute-sanitizer --version` | **2025.3.1.0** (ships with the toolkit) |
| `cmake --version` | 4.4.3 |
| MSVC `cl` | **19.44.35229** (VS 2022 Build Tools 17.14.41 + VCTools workload; installed 2026-10-03. NOTE: winget installed only the skeleton; the compiler required `vs_BuildTools.exe --add Microsoft.VisualStudio.Workload.VCTools` directly) |
| `g++ --version` | 16.1.0 (MSYS2 Rev5) |
| `clang-format --version` | **23.1.3** (via `pip install clang-format` in the Windows Python; pinned in CI) |
| `python --version` | 3.14.5 (MSYS2, no pip — repo scripts run here; Windows pythoncore 3.14 has pip/matplotlib/torch-CPU) |
| `git --version` | 2.55.0.windows.5 (repo initialized 2026-10-09; new bench JSONs stamp the short hash, older ones say `"unknown"`) |

Compiler/CMake used for this record: MSVC vs MSYS2 g++ depends on the shell;
the CMake CPU-only build works with either (verified below with the default generator).

## Profiler permission

Update 2026-10-15: counter collection WORKS. `ncu --query-metrics` succeeds
and `scripts/run_ncu_ladder.py` collected counters for v2/v4/v5/v6/cuBLAS at
1024³ (`results/ncu/ladder_M1024_N1024_K1024/`), run with
`--clock-control none` so ncu never locks clocks. No permission change was
made from this repository; it changed between sessions. Older notes below
are kept as history.

Update 2026-10-09: toolkit IS installed (`ncu`/`compute-sanitizer` 2025.3.1.0).
Counter permission was blocked at the 2026-10-09 attempt (`ERR_NVGPUCTRPERM`,
log in `results/ncu/`); re-tested after counters were enabled and STILL
blocked — even `ncu --query-metrics` fails with `ERR_NVGPUCTRPERM`
(`results/ncu/query_metrics_ERR_NVGPUCTRPERM.log`; section list saved in
`results/ncu/list_sections_2025.3.1.log`). Profiler evidence stays `NOT RUN` — rely on `cudaEvent` timing plus clearly labeled analytical
reasoning. (No admin/developer-mode change attempted — out of scope.)

Original note (kept for history): shared/cloud GPUs often block profiler
counters. This is a local WDDM GPU; WDDM + Nsight Compute generally works but
some counters need TCC/admin; record the outcome of `scripts/run_ncu.sh`.

## `-arch` decision

Default in `CMakeLists.txt`: `KERN_CUDA_ARCH=sm_120` (matches RTX 5070 inference).
Override per machine, e.g. `-DKERN_CUDA_ARCH=sm_90` (H100) / `sm_80` (A100).
Release flags: `-O3 -lineinfo` (+ arch). Never time a `-G` build.

## What this means (honesty gate)

- Update 2026-10-09: GPU execution **works on this machine now**. `hello.exe`
  (real CUDA: malloc + kernel + memcpy) prints 42, and `kern_correctness`
  runs to completion (ALL PASS, see `LOG.md`). The earlier SAC block
  (`VerifiedAndReputableDesktop`, events 3077/3118) no longer stops execution —
  and the pre-fix `kern_correctness` hang is attributed to the v3
  divergent-`__syncthreads` bug (fixed, see `LOG.md`), not to SAC.
- Original 2026-10-03 note (kept for history): GPU code compiled cleanly but
  could not execute: Smart App Control blocked freshly built unsigned
  test/bench binaries and no Defender cloud verdict was retrieved, so every
  GPU result remained `NOT RUN` until the binaries ran here or elsewhere.
- CPU reference, Python tests, claims audit, and table generation run here
  and are verified in `LOG.md`.

## Reproduce on a CUDA machine (Colab / Kaggle / local)

```bash
# 1. Clone + environment record
git clone <this-repo-url> kern && cd kern
nvidia-smi --query-gpu=name,driver_version --format=csv > /tmp/smi.txt; cat /tmp/smi.txt
nvcc --version
ncu --version; nsys --version; compute-sanitizer --version
python3 tools/detect_arch.py   # confirm sm_XXX, then use it below

# 2. Build (example: RTX 5070 -> sm_120; A100 -> sm_80)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKERN_CUDA_ARCH=sm_120
cmake --build build -j
ctest --test-dir build --output-on-failure

# 3. Correctness + sanitizer (save logs where the audit expects them)
./build/kern_correctness --seed 1234 --out results/correctness_seed1234.json
bash scripts/run_sanitizer.sh ./build/kern_correctness
bash scripts/run_ncu.sh 1024 1024 1024   # after confirming metric names per script header

# 4. Benchmarks (run EACH at least twice; flag >5% median drift before reporting)
./build/kern_bench --all --seed 0
./build/kern_bench --all --seed 0   # second run -> compare medians
python3 scripts/generate_tables.py --write
python3 scripts/claims_audit.py
```

Log files go to: `results/bench_*.json`, `results/correctness_*.json`,
`results/sanitizer/*.log`, `results/ncu/*`. Commit all of them.
