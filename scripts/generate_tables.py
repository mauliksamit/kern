#!/usr/bin/env python3
"""Generate README results table from results/*.json (never typed by hand).

Every number in the README comes from a file in
results/ produced by a script. This script reads bench JSONs and emits markdown.

Usage:
  python scripts/generate_tables.py            # print table to stdout
  python scripts/generate_tables.py --check    # exit 1 if README table is stale
  python scripts/generate_tables.py --write    # splice table between markers in README.md
"""
import glob
import json
import os
import sys

BEGIN = "<!-- RESULTS-BEGIN -->"
END = "<!-- RESULTS-END -->"
RESULTS_GLOB = os.path.join(os.path.dirname(__file__), "..", "results", "bench_*.json")


def load_rows():
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        with open(path) as f:
            d = json.load(f)
        for r in d.get("results", []):
            if r.get("failed", False):
                continue  # backend failure recorded in JSON; never tabulated
            rows.append(
                {
                    "shape": f"{d['M']}x{d['N']}x{d['K']}",
                    "kernel": r["kernel"],
                    "median_ms": r["median_ms"],
                    "gflops": r["gflops"],
                    "gbps": r["gbps"],
                    "file": os.path.basename(path),
                }
            )
    return rows


def render(rows):
    if not rows:
        return (
            "| Shape (MxNxK) | Kernel | Median (ms) | GFLOPS | GB/s | Source |\n"
            "|---|---|---|---|---|---|\n"
            "| *all GPU results* | *NOT RUN* | *NOT RUN* | *NOT RUN* | *NOT RUN* | "
            "_No results/*.json yet. Run the bench on a CUDA machine; see README._ |\n"
        )
    out = ["| Shape (MxNxK) | Kernel | Median (ms) | GFLOPS | GB/s | Source |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['shape']} | {r['kernel']} | {r['median_ms']:.4f} | "
            f"{r['gflops']:.1f} | {r['gbps']:.1f} | `{r['file']}` |"
        )
    return "\n".join(out) + "\n"


def main():
    rows = load_rows()
    table = render(rows)
    if "--write" in sys.argv:
        readme = os.path.join(os.path.dirname(__file__), "..", "README.md")
        with open(readme) as f:
            text = f.read()
        if BEGIN not in text or END not in text:
            print("README markers missing", file=sys.stderr)
            return 1
        pre, _, rest = text.partition(BEGIN)
        _, _, post = rest.partition(END)
        new = pre + BEGIN + "\n" + table + END + post
        with open(readme, "w") as f:
            f.write(new)
        print(f"README table updated from {len(rows)} rows.")
        return 0
    if "--check" in sys.argv:
        readme = os.path.join(os.path.dirname(__file__), "..", "README.md")
        with open(readme) as f:
            text = f.read()
        if table.strip() not in text:
            print("README results table is stale. Run: python scripts/generate_tables.py --write",
                  file=sys.stderr)
            return 1
        print("README table is up to date.")
        return 0
    sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
