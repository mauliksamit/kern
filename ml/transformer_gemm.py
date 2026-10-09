#!/usr/bin/env python3
"""Transformer GEMM micro-benchmarks linking torch to the kern CUDA ladder.

Measures representative transformer-layer GEMMs (tokens x d_model against
d_model / FFN width / QKV projections) with torch.matmul (eager), optionally
torch.compile, and reports them next to this repo's best custom kernel and the
cuBLAS baseline measured by the C++ harness on IDENTICAL shapes
(results/ml_M*_N*_K*.json, produced by kern_bench --M/--N/--K).

Protocol mirrors src/bench.cu: FP32, TF32 explicitly disabled for the FP32
rows (torch.backends.cuda.matmul.allow_tf32 = False, asserted), CUDA events
around the op only, >=10 warmup + >=100 timed iterations, median/min/p95 over
identical inputs (torch.manual_seed fixed). A clearly labeled TF32-allowed row
is reported separately, never mixed into the FP32 comparison.

Numerics: FP32 eager output vs float64 reference (max rel err).

torch.compile may be unsupported on native Windows (needs Triton). Failure is
recorded verbatim in the JSON + docs/NEGATIVE_RESULTS.md and marked NOT RUN;
no workarounds attempted.

Usage (from repo root, with the ml venv):
  ml\\.venv\\Scripts\\python.exe ml/transformer_gemm.py --out results/ml_transformer.json
Requires: CUDA torch (see Phase 4 LOG). CPU-only torch exits non-zero (NOT RUN).
"""
import json
import os
import statistics
import sys

SHAPES = [
    # (tag, M tokens, K in-features, N out-features)
    ("attn-proj-512", 512, 4096, 4096),
    ("attn-proj-2048", 2048, 4096, 4096),
    ("ffn-up-512", 512, 4096, 11008),
    ("ffn-down-512", 512, 11008, 4096),
    ("ffn-up-2048", 2048, 4096, 11008),
]

WARMUP = 10
ITERS = 100


def need_cuda():
    import torch

    if not torch.cuda.is_available():
        print("CUDA torch required; this build is CPU-only. Marking Phase 4 NOT RUN.")
        return None
    return torch


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    n = len(samples)
    return {
        "median_ms": samples[n // 2],
        "min_ms": samples[0],
        "p95_ms": samples[int(0.95 * (n - 1))],
        "mean_ms": sum(samples) / n,
        "iters": samples,
    }


def cpp_reference(tag, M, K, N):
    """Best custom kernel + cuBLAS from the C++ harness JSON for this shape."""
    path = os.path.join("results", f"ml_M{M}_N{N}_K{K}.json")
    if not os.path.exists(path):
        return {"note": f"missing {path}; run kern_bench --M {M} --N {N} --K {K} first"}
    with open(path) as f:
        d = json.load(f)
    customs = [r for r in d.get("results", [])
               if not r.get("failed", False) and not r.get("skipped", False)
               and r["kernel"] != "cublas"]
    blas = [r for r in d.get("results", []) if r["kernel"] == "cublas"
            and not r.get("failed", False)]
    out = {}
    if customs:
        best = max(customs, key=lambda r: r["gflops"])
        out["best_custom"] = {"kernel": best["kernel"],
                              "median_ms": best["median_ms"],
                              "gflops": best["gflops"]}
    if blas:
        out["cublas"] = {"median_ms": blas[0]["median_ms"],
                         "gflops": blas[0]["gflops"]}
    out["source"] = os.path.basename(path)
    return out


def main():
    out_path = "results/ml_transformer.json"
    for i, a in enumerate(sys.argv[1:]):
        if a == "--out" and i + 1 < len(sys.argv[1:]):
            out_path = sys.argv[1:][i + 1]
    torch = need_cuda()
    if torch is None:
        return 3
    print(f"torch {torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)}")

    rows = []
    for tag, M, K, N in SHAPES:
        print(f"[{tag}] M={M} K={K} N={N}")
        torch.manual_seed(0)
        A = (torch.rand(M, K, dtype=torch.float32, device="cuda") * 2 - 1)
        B = (torch.rand(K, N, dtype=torch.float32, device="cuda") * 2 - 1)

        # FP32 rows: TF32 explicitly off (asserted, recorded).
        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        row = {"tag": tag, "M": M, "K": K, "N": N, "dtype": "float32",
               "tf32": False}
        eager = lambda: torch.matmul(A, B)  # noqa: E731
        t = time_fn(eager)
        C = eager()
        torch.cuda.synchronize()
        t["gflops"] = (2.0 * M * N * K) / (t["median_ms"] / 1e3) / 1e9
        row["eager"] = t

        # Numerics vs float64 reference.
        with torch.no_grad():
            ref = torch.matmul(A.double(), B.double())
            got = C.double()
            denom = ref.abs().max().item() or 1.0
            row["numerics"] = {
                "max_abs_err": (got - ref).abs().max().item(),
                "max_rel_err": ((got - ref).abs() / denom),
            }
            row["numerics"]["max_rel_err"] = row["numerics"]["max_rel_err"].max().item() \
                if hasattr(row["numerics"]["max_rel_err"], "max") else \
                float(row["numerics"]["max_rel_err"])

        # torch.compile on the same op (may fail on Windows: recorded, NOT RUN).
        try:
            compiled = torch.compile(lambda x, y: torch.matmul(x, y))
            # warmup triggers compilation; extra warmups absorb recompile
            for _ in range(3):
                compiled(A, B)
            torch.cuda.synchronize()
            tc = time_fn(lambda: compiled(A, B))
            tc["gflops"] = (2.0 * M * N * K) / (tc["median_ms"] / 1e3) / 1e9
            row["compiled"] = tc
            print(f"  eager {t['median_ms']:.4f} ms ({t['gflops']:.1f} GFLOPS), "
                  f"compiled {tc['median_ms']:.4f} ms ({tc['gflops']:.1f} GFLOPS)")
        except Exception as e:  # noqa: BLE001 - record verbatim per protocol
            row["compiled"] = {"NOT RUN": f"{type(e).__name__}: {e}"}
            print(f"  eager {t['median_ms']:.4f} ms ({t['gflops']:.1f} GFLOPS), "
                  f"compiled NOT RUN ({type(e).__name__})")

        # Clearly labeled TF32-allowed row (separate comparison only).
        torch.backends.cuda.matmul.allow_tf32 = True
        t32 = time_fn(eager)
        t32["gflops"] = (2.0 * M * N * K) / (t32["median_ms"] / 1e3) / 1e9
        row["tf32_allowed"] = t32
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"  tf32-allowed {t32['median_ms']:.4f} ms ({t32['gflops']:.1f} GFLOPS) [separate]")

        row["cpp"] = cpp_reference(tag, M, K, N)
        rows.append(row)

    doc = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "tf32_default": False,
        "warmup": WARMUP,
        "iters": ITERS,
        "protocol": "CUDA events around op only; median/min/p95; seed 0",
        "shapes": rows,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"wrote {out_path} ({len(rows)} shapes)")
    # median GFLOPS sanity for the console record
    med = statistics.median(r["eager"]["gflops"] for r in rows)
    print(f"median eager GFLOPS across shapes: {med:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
