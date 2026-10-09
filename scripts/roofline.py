#!/usr/bin/env python3
"""Roofline-style placement of a GEMM shape against measured ladder data.

Given M/N/K (e.g. a transformer projection: M = batch*tokens, N/K = width),
prints FLOPs, ideal traffic, arithmetic intensity, the nearest measured ladder
shape and its best-custom GFLOPS, and the ideal-traffic bandwidth that GFLOPS
implies. All performance numbers come from results/bench_*.json; the intensity
math is analytical (documented below), not a measurement.

  AI = 2*M*N*K / ((M*K + K*N + M*N) * 4)   [FLOP per ideal byte, FP32]
  implied GB/s = GFLOPS_measured / AI       [lower bound on actual traffic rate:
                                            real kernels move MORE than ideal]

Usage:
  python3 scripts/roofline.py --M 512 --N 4096 --K 4096 --label ffn-up-proj
  python3 scripts/roofline.py --self-check   # validates math on 1024^3 vs JSON
  python3 scripts/roofline.py --ml results/ml_transformer.json
      # place each MEASURED torch shape at its AI with its measured GFLOPS
      # (analytical model: AI math + nearest-ladder reference, labeled as such)
"""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_GLOB = os.path.join(ROOT, "results", "bench_*.json")
CUSTOM = ["v0_naive", "v1_coalesced", "v2_tiled", "v3_tiled_vectorized", "v4_coarsened",
          "v5_regtile", "v6_vec_transposed"]


def stats(M, N, K):
    flops = 2.0 * M * N * K
    ideal_bytes = (float(M * K) + float(K * N) + float(M * N)) * 4.0
    return flops, ideal_bytes, flops / ideal_bytes


def nearest(M, N, K):
    best, best_key = None, None
    for path in sorted(glob.glob(RESULTS_GLOB)):
        with open(path) as f:
            d = json.load(f)
        dist = abs(d["M"] - M) + abs(d["N"] - N) + abs(d["K"] - K)
        if best is None or dist < best[0]:
            best = (dist, path, d)
    _, path, d = best
    rows = [r for r in d.get("results", [])
            if not r.get("failed", False) and r["kernel"] in CUSTOM]
    top = max(rows, key=lambda r: r["gflops"])
    return os.path.basename(path), (d["M"], d["N"], d["K"]), top


def report(M, N, K, label):
    flops, ideal_bytes, ai = stats(M, N, K)
    print(f"[{label}] M={M} N={N} K={K}")
    print(f"  FLOPs: {flops / 1e9:.3f} GFLOP, ideal traffic: {ideal_bytes / 1e6:.2f} MB, "
          f"AI: {ai:.1f} FLOP/B")
    try:
        path, shape, top = nearest(M, N, K)
    except (ValueError, KeyError, FileNotFoundError):
        print("  no bench JSONs yet (NOT RUN) — no measured reference available.")
        return 0
    print(f"  nearest measured shape: {shape[0]}x{shape[1]}x{shape[2]} ({path})")
    print(f"  best custom kernel there: {top['kernel']} at {top['gflops']:.1f} GFLOPS "
          f"(median {top['median_ms']:.4f} ms)")
    print(f"  implied ideal-traffic rate: {top['gflops'] / ai:.1f} GB/s "
          f"(lower bound; real traffic is higher — compare GB/s column in README)")
    return 0


def self_check():
    # 1024^3: AI = 2*1024/12 = 170.67 FLOP/B by hand; implied BW from the
    # committed v4 number must reproduce the committed GB/s column.
    flops, ideal, ai = stats(1024, 1024, 1024)
    assert abs(ai - 2 * 1024 / 12) < 1e-9, ai
    d = json.load(open(os.path.join(ROOT, "results", "bench_M1024_N1024_K1024.json")))
    v4 = [r for r in d["results"] if r["kernel"] == "v4_coarsened"][0]
    implied = v4["gflops"] / ai
    assert abs(implied - v4["gbps"]) < 0.5, (implied, v4["gbps"])
    print(f"self-check OK: AI=170.67 FLOP/B, v4 implied {implied:.1f} GB/s "
          f"== committed {v4['gbps']:.1f} GB/s")
    return 0


def report_ml(path):
    """Place each measured torch shape at its AI with measured GFLOPS.

    Analytical model (labeled as such): AI from dimensions; the torch GFLOPS
    are measured (ml JSON); the nearest-ladder reference is measured (bench
    JSONs). Only the intensity arithmetic is analytical.
    """
    with open(path) as f:
        doc = json.load(f)
    for s in doc.get("shapes", []):
        M, K, N = s["M"], s["K"], s["N"]
        flops, ideal_bytes, ai = stats(M, N, K)
        g = s.get("eager", {}).get("gflops", float("nan"))
        print(f"[{s['tag']}] M={M} K={K} N={N} AI={ai:.1f} FLOP/B, "
              f"torch-eager {g:.1f} GFLOPS -> implied {g / ai:.1f} GB/s ideal")
        try:
            bpath, shape, top = nearest(M, N, K)
        except (ValueError, KeyError, FileNotFoundError):
            print("  no bench JSONs (NOT RUN)")
            continue
        print(f"  nearest ladder: {shape[0]}x{shape[1]}x{shape[2]} {top['kernel']} "
              f"{top['gflops']:.1f} GFLOPS ({bpath})")
    return 0


def main():
    if "--self-check" in sys.argv:
        return self_check()
    if "--ml" in sys.argv:
        i = sys.argv.index("--ml")
        if i + 1 >= len(sys.argv):
            print(__doc__)
            return 2
        return report_ml(sys.argv[i + 1])
    args = {"--label": "gemm"}
    for i, a in enumerate(sys.argv[1:]):
        if a.startswith("--") and i + 1 < len(sys.argv[1:]):
            args[a] = sys.argv[1:][i + 1]
    try:
        M, N, K = int(args["--M"]), int(args["--N"]), int(args["--K"])
    except KeyError:
        print(__doc__)
        return 2
    return report(M, N, K, args["--label"])


if __name__ == "__main__":
    sys.exit(main())
