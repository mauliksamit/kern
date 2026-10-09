#!/usr/bin/env python3
"""Old-vs-new median comparison for the fast-math removal experiment.

Compares bench JSONs in two directories (e.g. fast-math baseline extracted
from git vs fresh strict-IEEE runs) per (shape, kernel) median. Prints a
markdown table for docs/WRITEUP.md and flags any kernel changing by more
than 5% at any shape.

Usage:
  python3 scripts/compare_runs.py <old_dir> <new_dir>   # e.g. /tmp/fm_old results
"""
import glob
import json
import os
import sys


def load(dirpath):
    data = {}
    for path in sorted(glob.glob(os.path.join(dirpath, "bench_*.json"))):
        with open(path) as f:
            d = json.load(f)
        shape = (d["M"], d["N"], d["K"])
        for r in d.get("results", []):
            if r.get("failed", False) or r.get("skipped", False):
                continue
            data[(shape, r["kernel"])] = (r["median_ms"], r["gflops"])
    return data


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old, new = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(old) & set(new))
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    print("| Shape | Kernel | Old median (ms) | New median (ms) | Change | >5%? |")
    print("|---|---|---|---|---|---|")
    big = []
    for k in keys:
        m1, g1 = old[k]
        m2, g2 = new[k]
        pct = (m2 - m1) / m1 * 100 if m1 else float("inf")
        flag = "YES" if abs(pct) > 5 else "no"
        if abs(pct) > 5:
            big.append((k, pct))
        shape = f"{k[0][0]}x{k[0][1]}x{k[0][2]}"
        print(f"| {shape} | {k[1]} | {m1:.4f} | {m2:.4f} | {pct:+.2f}% | {flag} |")
    for k in sorted(only_old):
        print(f"| {k[0]} | {k[1]} | {old[k][0]:.4f} | — (missing) | — | — |")
    for k in sorted(only_new):
        print(f"| {k[0]} | {k[1]} | — (missing) | {new[k][0]:.4f} | — | — |")
    print(f"\ncompared {len(keys)} cells; {len(big)} above 5%.")
    for k, pct in big:
        print(f"  >5%: {k[0]} {k[1]} {pct:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
