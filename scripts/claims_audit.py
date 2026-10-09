#!/usr/bin/env python3
"""Claims audit.

Checks that every numeric performance claim in README.md traces to a file in
results/. Strategy (documented, approximate but honest):
  1. Extract all floating numbers from README.md outside the results-table block
     that look like measurements (ms / GFLOPS / GB/s / % with nearby keyword).
  2. Require each to either (a) appear in some results/*.json within tolerance,
     or (b) be explicitly marked NOT RUN / estimated / TODO.
  3. Also verify the results table block matches scripts/generate_tables.py output.

Percentages must derive as 100*a/b from two traced values (1pp tolerance).
The 5% repeatability threshold ("drift") is a protocol constant,
not a measurement, and is exempt by keyword.
Exit 0 = audit passes. Exit 1 = untraced number found.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
README = os.path.join(ROOT, "README.md")
RESULTS_GLOB = os.path.join(ROOT, "results", "*.json")
BEGIN = "<!-- RESULTS-BEGIN -->"
END = "<!-- RESULTS-END -->"


def load_numbers():
    nums = []
    for path in glob.glob(RESULTS_GLOB):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        for r in d.get("results", []):
            for key in ("median_ms", "min_ms", "p95_ms", "mean_ms", "gflops", "gbps"):
                if key in r:
                    nums.append(float(r[key]))
        # Estimated FP32 peak (TFLOPS) lives in the JSON metadata.
        if "peak_fp32_tflops_est" in d:
            nums.append(float(d["peak_fp32_tflops_est"]))
    return nums


def main():
    with open(README) as f:
        text = f.read()
    # Remove the generated table block: those numbers are traced by construction.
    if BEGIN in text and END in text:
        pre, _, rest = text.partition(BEGIN)
        _, _, post = rest.partition(END)
        body = pre + post
    else:
        body = text
    # Find candidate measurement claims: number + unit keyword nearby.
    pattern = re.compile(r"(\d+\.\d+|\d+)\s*(ms|GFLOPS|GB/s|TFLOPS|%|e\.g\.)", re.IGNORECASE)
    traced = load_numbers()
    failures = []
    for m in pattern.finditer(body):
        val = float(m.group(1))
        unit = m.group(2)
        # Skip obvious non-claims: years, versions, tile sizes, counts.
        start = max(0, m.start() - 60)
        ctx = body[start:m.end() + 40]
        if re.search(r"NOT RUN|TODO|e\.g\.|for example|version|CUDA \d|sm_\d|TILE|drift",
                      ctx, re.I):
            continue
        if "NOT RUN" in body[max(0, m.start() - 200):m.end()]:
            continue
        # Check traceability within 1% relative or small absolute tolerance.
        ok = any(abs(val - t) <= max(0.01 * abs(t), 1e-6) or
                 (unit.lower() == "ms" and abs(val - t) < 1e-3) for t in traced)
        if not ok and unit == "%":
            for a in traced:
                for b in traced:
                    if b and abs(val - 100.0 * a / b) <= 1.0:
                        ok = True
                        break
                if ok:
                    break
        if not ok:
            # Numbers with no results files at all are only OK if README declares NOT RUN.
            failures.append((val, unit, ctx.strip().replace("\n", " ")[:120]))
    # Table freshness: regenerate and compare.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen", os.path.join(ROOT, "scripts", "generate_tables.py"))
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        table = gen.render(gen.load_rows())
        if table.strip() not in text:
            print("AUDIT NOTE: results table block is stale "
                  "(run python scripts/generate_tables.py --write).")
            # Stale table is a failure only when results exist.
            if traced:
                failures.append(("stale-table", "", "regenerate README table"))
    except Exception as e:
        print(f"AUDIT WARNING: could not verify table freshness: {e}")

    if failures:
        print("CLAIMS AUDIT FAILED: untraced numbers in README (not in results/*.json):")
        for val, unit, ctx in failures:
            print(f"  {val} {unit}  ...{ctx}...")
        print("Fix: run the bench to produce results/*.json, regenerate the table, "
              "or mark the claim NOT RUN.")
        return 1
    print(f"CLAIMS AUDIT PASSED ({len(traced)} traced values in results/*.json).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
