#!/usr/bin/env python3
"""Plot median GFLOPS vs shape per kernel from results/bench_*.json.

The figure is generated from committed bench JSONs,
never drawn by hand. Requires matplotlib (not needed by any other script or CI;
on this Windows machine run with the Python that has it, e.g. the
pythoncore 3.14 install).

Usage:
  python3 scripts/plot_results.py            # writes results/plot_gflops.png
"""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_GLOB = os.path.join(ROOT, "results", "bench_*.json")
OUT = os.path.join(ROOT, "results", "plot_gflops.png")

ORDER = ["v0_naive", "v1_coalesced", "v2_tiled", "v3_tiled_vectorized",
         "v4_coarsened", "v5_regtile", "cublas"]


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot "
              "(pip install matplotlib, then re-run).")
        return 0
    shapes, series = [], {k: [] for k in ORDER}
    for path in sorted(glob.glob(RESULTS_GLOB)):
        with open(path) as f:
            d = json.load(f)
        label = f"{d['M']}x{d['N']}x{d['K']}"
        shapes.append(label)
        vals = {r["kernel"]: r["gflops"] for r in d.get("results", [])
                if not r.get("failed", False)}
        for k in ORDER:
            series[k].append(vals.get(k, float("nan")))
    if not shapes:
        print("no results/bench_*.json; nothing to plot (NOT RUN).")
        return 0
    x = range(len(shapes))
    width = 0.8 / len(ORDER)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(ORDER):
        offs = [v + (i - len(ORDER) / 2 + 0.5) * width for v in x]
        ax.bar(offs, series[k], width=width, label=k)
    ax.set_xticks(list(x), shapes, rotation=15)
    ax.set_ylabel("Median GFLOPS (2*M*N*K / median time)")
    ax.set_title("kern SGEMM ladder — median GFLOPS per shape (RTX 5070, sm_120)")
    ax.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT} from {len(shapes)} shapes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
