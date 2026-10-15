#!/usr/bin/env python3
"""Generate a summary (JSON + Markdown table) from run_ncu_ladder.py CSVs.

No GPU needed. Reads results/ncu/ladder_*/<kernel>.csv (ncu --page raw) and
writes summary.json + summary.md next to them. Every value is copied from the
CSV; derived ratios are defined below and labeled "derived" in the output.

Derived quantities (definitions):
- shared_ld_wavefronts_per_inst = l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
                                  / smsp__sass_inst_executed_op_shared_ld.sum
  (shared-memory passes per warp-level load instruction; 1.0 is the minimum
  for a 32-bit load, a 128-bit load needs more passes even without conflicts,
  so compare like-for-like kernels only)
- shared_st_wavefronts_per_inst: same for stores.
- ffma_thread_inst_over_MNK = smsp__sass_thread_inst_executed_op_ffma_pred_on.sum / (M*N*K)
  (1.0 means one thread-level FFMA per multiply-add of the GEMM)

Usage: python scripts/ncu_summary.py [results/ncu/ladder_M1024_N1024_K1024]
"""
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORDER = ["v2_tiled", "v4_coarsened", "v5_regtile", "v6_vec_transposed", "cublas"]
# (csv column, short label, unit)
ROWS = [
    ("launch__block_size", "threads/block", ""),
    ("launch__grid_size", "blocks", ""),
    ("launch__registers_per_thread", "registers/thread", ""),
    ("launch__shared_mem_per_block_static", "static shared/block", "B"),
    ("launch__shared_mem_per_block_dynamic", "dynamic shared/block", "B"),
    ("launch__occupancy_limit_registers", "block limit: registers", "blocks/SM"),
    ("launch__occupancy_limit_shared_mem", "block limit: shared mem", "blocks/SM"),
    ("launch__occupancy_limit_warps", "block limit: warps", "blocks/SM"),
    ("sm__maximum_warps_per_active_cycle_pct", "theoretical occupancy", "%"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy", "%"),
    ("launch__waves_per_multiprocessor", "waves per SM", ""),
    ("smsp__inst_executed.sum", "warp instructions executed", ""),
    ("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "thread FFMA instructions", ""),
    ("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", "FMA pipe active", "%"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput (max sub-unit)", "%"),
    ("smsp__sass_inst_executed_op_shared_ld.sum", "shared load instructions", ""),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "shared load wavefronts", ""),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "shared load bank conflicts", ""),
    ("smsp__sass_inst_executed_op_shared_st.sum", "shared store instructions", ""),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", "shared store wavefronts", ""),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "shared store bank conflicts", ""),
    ("dram__bytes_op_read.sum", "DRAM bytes read", "B"),
    ("dram__bytes_op_write.sum", "DRAM bytes written", "B"),
    ("lts__t_bytes.sum", "L2 bytes", "B"),
    ("gpu__time_duration.sum", "duration under ncu (clocks unlocked; not a speed claim)", "ns"),
]


def num(s):
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "ncu", "ladder_M1024_N1024_K1024")
    d = os.path.join(ROOT, d) if not os.path.isabs(d) else d
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    M, N, K = meta["shape"]["M"], meta["shape"]["N"], meta["shape"]["K"]
    out = {"source_dir": os.path.relpath(d, ROOT).replace("\\", "/"), "git_commit": meta["git_commit"],
           "ncu_version": meta["ncu_version"], "gpu": meta["gpu"], "shape": meta["shape"],
           "kernels": {}}
    for k in ORDER:
        path = os.path.join(d, f"{k}.csv")
        if not os.path.exists(path):
            out["kernels"][k] = {"status": "NOT RUN (no CSV)"}
            continue
        rows = list(csv.reader(open(path, encoding="utf-8")))
        h, v = rows[0], rows[2]
        rec = {"kernel_name": v[h.index("Kernel Name")], "metrics": {}, "derived": {}}
        for col, _, _ in ROWS:
            rec["metrics"][col] = num(v[h.index(col)]) if col in h else None
        m = rec["metrics"]

        def ratio(a, b):
            return round(m[a] / m[b], 3) if m.get(a) is not None and m.get(b) else None

        rec["derived"]["shared_ld_wavefronts_per_inst"] = ratio(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "smsp__sass_inst_executed_op_shared_ld.sum")
        rec["derived"]["shared_st_wavefronts_per_inst"] = ratio(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", "smsp__sass_inst_executed_op_shared_st.sum")
        f = m.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
        rec["derived"]["ffma_thread_inst_over_MNK"] = round(f / (M * N * K), 4) if f else None
        out["kernels"][k] = rec

    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")

    def fmt(x):
        if x is None:
            return "n/a"
        return f"{x:,.2f}" if x != int(x) else f"{int(x):,}"

    ks = [k for k in ORDER if "metrics" in out["kernels"][k]]
    lines = [f"# Nsight Compute ladder profile, {M}x{N}x{K} (generated by scripts/ncu_summary.py)", "",
             f"Source: `{out['source_dir']}/*.csv` (ncu {out['ncu_version']}, commit `{out['git_commit'][:7]}`, "
             f"GPU {out['gpu']}). One profiled launch per kernel (first timed iteration), "
             "`--clock-control none`. Durations under ncu are not speed claims; "
             "speed comes from the cudaEvent bench JSONs.", "",
             "| Metric | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
    for col, label, unit in ROWS:
        u = f" ({unit})" if unit else ""
        lines.append(f"| {label}{u} `{col}` | " + " | ".join(fmt(out["kernels"][k]["metrics"][col]) for k in ks) + " |")
    for dk in ("shared_ld_wavefronts_per_inst", "shared_st_wavefronts_per_inst", "ffma_thread_inst_over_MNK"):
        lines.append(f"| derived: {dk} | " + " | ".join(fmt(out["kernels"][k]["derived"][dk]) for k in ks) + " |")
    lines += ["", "cuBLAS kernel: `" + out["kernels"].get("cublas", {}).get("kernel_name", "n/a") + "`", ""]
    with open(os.path.join(d, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {out['source_dir']}/summary.json and summary.md")


if __name__ == "__main__":
    main()
