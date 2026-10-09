#!/usr/bin/env python3
"""Repeat-run drift check.

Compares median kernel times between two bench runs over identical shapes and
flags any (shape, kernel) whose medians differ by more than 5%. Because
`kern_bench --all` rewrites the same `results/bench_*.json` paths, preserve the
first run's files in a separate directory before the second run:

  ./build/kern_bench --all --seed 0
  mkdir -p /tmp/kern_run1 && cp results/bench_*.json /tmp/kern_run1/
  ./build/kern_bench --all --seed 0
  python3 scripts/check_repeat.py /tmp/kern_run1 results

Exit 0 = all drifts within 5%. Exit 1 = drift above threshold (investigate
before reporting) or incomparable file sets.
"""
import glob
import json
import os
import sys

THRESHOLD = 0.05


def load(dirpath):
    data = {}
    for path in sorted(glob.glob(os.path.join(dirpath, "bench_*.json"))):
        with open(path) as f:
            d = json.load(f)
        shape = (d["M"], d["N"], d["K"])
        for r in d.get("results", []):
            if r.get("failed", False):
                continue
            data[(shape, r["kernel"])] = r["median_ms"]
    return data


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    if set(a) != set(b):
        only_a = set(a) - set(b)
        only_b = set(b) - set(a)
        print(f"INCOMPARABLE: run1={len(a)} entries, run2={len(b)} entries")
        for k in sorted(only_a):
            print(f"  only in run1: {k}")
        for k in sorted(only_b):
            print(f"  only in run2: {k}")
        return 1
    worst = 0.0
    bad = []
    for k in sorted(a):
        m1, m2 = a[k], b[k]
        drift = abs(m2 - m1) / m1 if m1 else float("inf")
        worst = max(worst, drift)
        flag = "OK " if drift <= THRESHOLD else "DRIFT"
        if drift > THRESHOLD:
            bad.append(k)
        print(f"  [{flag}] shape={k[0]} kernel={k[1]:18s} "
              f"run1={m1:.4f}ms run2={m2:.4f}ms drift={100*drift:.2f}%")
    if bad:
        print(f"DRIFT CHECK FAILED: {len(bad)} entries above "
              f"{100*THRESHOLD:.0f}% (worst {100*worst:.2f}%). Investigate before reporting.")
        return 1
    print(f"DRIFT CHECK PASSED ({len(a)} entries, worst drift {100*worst:.2f}%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
