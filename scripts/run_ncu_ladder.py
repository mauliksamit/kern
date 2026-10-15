#!/usr/bin/env python3
"""Per-kernel Nsight Compute profile of the ladder at one shape.

Profiles v2, v4, v5, v6 and the cuBLAS baseline inside kern_bench (identical
inputs and launch configs as the reported benchmarks) and writes, per kernel,
a raw-page CSV plus the console log into results/ncu/ladder_M{M}_N{N}_K{K}/,
and a meta.json with provenance (commit, ncu version, driver, command lines).
Summarize with scripts/ncu_summary.py (no GPU needed for that step).

Rules followed:
- Every --metrics name below was confirmed present in `ncu --query-metrics`
  for ncu 2025.3.1 (results/ncu/query_metrics_2025.3.1.txt). Rollup suffixes
  (.sum/.avg/.pct_of_peak_...) are ncu's standard rollups of those base names.
  Launch/occupancy values come from the LaunchStats and Occupancy sections,
  confirmed by `ncu --list-sections` (results/ncu/list_sections_2025.3.1.log).
- --clock-control none: ncu's default locks GPU clocks to base during
  profiling; this repo never changes clock state. Consequence: durations under
  ncu follow the laptop's boost behaviour and are NOT used for speed claims
  (cudaEvent bench JSONs are). Counts (instructions, wavefronts, conflicts,
  bytes) do not depend on clock.
- --launch-skip 11 skips each kernel's 10 warmup launches + 1 probe launch in
  src/bench.cu; the profiled launch is the first timed iteration.

Usage: python scripts/run_ncu_ladder.py [--M 1024 --N 1024 --K 1024] [--bin build/kern_bench]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "lts__t_bytes.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
]
SECTIONS = ["LaunchStats", "Occupancy"]

# name -> ncu kernel filter. The cuBLAS kernel at 1024^3 was identified by a
# discovery run (logged in LOG.md 2026-10-15): cutlass SIMT sgemm "Kernel2<...>".
KERNELS = [
    ("v2_tiled", "regex:v2_kernel"),
    ("v4_coarsened", "regex:v4_kernel"),
    ("v5_regtile", "regex:v5_kernel"),
    ("v6_vec_transposed", "regex:v6_kernel"),
    ("cublas", "regex:Kernel2"),
]


def find_ncu():
    p = shutil.which("ncu")
    if p:
        return p
    # ncu.bat forwards args through cmd.exe, which mangles regex '|' etc.;
    # call the real executable directly on Windows.
    hits = glob.glob(r"C:\Program Files\NVIDIA Corporation\Nsight Compute *\target\*\ncu.exe")
    return sorted(hits)[-1] if hits else None


def redact(text):
    """Strip local absolute paths (repo root, home dir) from committed evidence."""
    for path, tag in ((ROOT, "<repo>"), (os.path.expanduser("~"), "<home>")):
        for variant in {path, path.replace("\\", "/")}:
            text = text.replace(variant, tag)
    return text


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--bin", default=os.path.join("build", "kern_bench"))
    a = ap.parse_args()

    ncu = find_ncu()
    if not ncu:
        print("ncu NOT FOUND: profiler evidence stays NOT RUN.")
        return 0
    binp = a.bin if os.path.exists(os.path.join(ROOT, a.bin)) else a.bin + ".exe"
    tag = f"M{a.M}_N{a.N}_K{a.K}"
    outdir = os.path.join(ROOT, "results", "ncu", f"ladder_{tag}")
    os.makedirs(outdir, exist_ok=True)
    scratch_json = os.path.join(outdir, "_bench_under_ncu.json")  # deleted below

    commit = run(["git", "rev-parse", "HEAD"]).stdout.strip() or "unknown"
    dirty = bool(run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.strip())
    ncu_ver = run([ncu, "--version"]).stdout.strip().splitlines()[-1:]
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"])
    meta = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "tracked_files_dirty": dirty,
        "ncu_version": ncu_ver[0] if ncu_ver else "unknown",
        "gpu": smi.stdout.strip(),
        "shape": {"M": a.M, "N": a.N, "K": a.K},
        "binary": binp,
        "clock_control": "none (clocks never locked; durations not used for claims)",
        "launch_skip": 11,
        "metrics": METRICS,
        "sections": SECTIONS,
        "kernels": {},
    }
    failed = False
    for name, filt in KERNELS:
        cmd = [ncu, "--clock-control", "none", "--csv", "--page", "raw",
               "-k", filt, "--launch-skip", "11", "-c", "1",
               "--metrics", ",".join(METRICS)]
        for s in SECTIONS:
            cmd += ["--section", s]
        cmd += [binp, "--M", str(a.M), "--N", str(a.N), "--K", str(a.K),
                "--seed", "0", "--out", scratch_json]
        print(f"[ncu] {name}: {filt}", flush=True)
        r = run(cmd)
        # ncu prints ==PROF== lines and the program's own stdout before the CSV.
        lines = r.stdout.splitlines()
        csv_start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
        log = "# command: ncu " + " ".join(cmd[1:]) + "\n"
        log += "\n".join(lines[:csv_start] if csv_start is not None else lines) + "\n" + r.stderr
        with open(os.path.join(outdir, f"{name}.log"), "w", encoding="utf-8") as f:
            f.write(redact(log))
        ok = r.returncode == 0 and csv_start is not None and "ERR_NVGPUCTRPERM" not in r.stdout + r.stderr
        if ok:
            with open(os.path.join(outdir, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("\n".join(lines[csv_start:]) + "\n")
        else:
            failed = True
            print(f"[ncu] {name} FAILED (exit {r.returncode}); see {name}.log", flush=True)
        meta["kernels"][name] = {"filter": filt, "ok": ok, "exit_code": r.returncode,
                                 "command": redact(" ".join(["ncu"] + cmd[1:]))}
    if os.path.exists(scratch_json):
        os.remove(scratch_json)  # bench timings under the profiler are not evidence
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        f.write(redact(json.dumps(meta, indent=2)) + "\n")
    print(f"Wrote {outdir}. Next: python scripts/ncu_summary.py {os.path.relpath(outdir, ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
