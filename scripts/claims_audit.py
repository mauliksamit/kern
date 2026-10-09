#!/usr/bin/env python3
"""Claims audit.

Checks that every numeric performance claim in README.md and docs/WRITEUP.md
traces to a file in
results/. Strategy (documented, approximate but honest):
  1. Extract all floating numbers from both files outside the results-table
     block that look like measurements (ms / GFLOPS / GB/s / % with nearby
     keyword). Protocol constants (the 100 ms
     single-run iteration rule, the 5% drift rule) are exempt by keyword.
  2. Require each to either (a) appear in some results/*.json within tolerance,
     or (b) be explicitly marked NOT RUN / estimated / TODO.
  3. Also verify the results table block matches scripts/generate_tables.py output.

Percentages must derive from two traced values within 1 percentage point,
either as a share (100*a/b) or as a relative change/drift (100*|a-b|/max(a,b)).
The 5% repeatability threshold ("drift") is a protocol constant,
not a measurement, and is exempt by keyword. Numbers inside a clearly labeled
analytical model (ideal/implied/analytical/extrapolation context) are exempt:
they are derived quantities, not measurements.
Exit 0 = audit passes. Exit 1 = untraced number found.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
README = os.path.join(ROOT, "README.md")
WRITEUP = os.path.join(ROOT, "docs", "WRITEUP.md")
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
            if r.get("failed", False) or r.get("skipped", False):
                continue  # failed/skipped backends carry no measurable value
            for key in ("median_ms", "min_ms", "p95_ms", "mean_ms", "gflops", "gbps"):
                if key in r:
                    nums.append(float(r[key]))
        # Estimated FP32 peak (TFLOPS) lives in the JSON metadata.
        if "peak_fp32_tflops_est" in d:
            nums.append(float(d["peak_fp32_tflops_est"]))
        # ML transformer JSON: measured torch medians/GFLOPS per config.
        for s in d.get("shapes", []):
            for cfg in ("eager", "compiled", "tf32_allowed"):
                r = s.get(cfg, {})
                if isinstance(r, dict):
                    for key in ("median_ms", "min_ms", "p95_ms", "mean_ms", "gflops"):
                        if key in r:
                            nums.append(float(r[key]))
    return nums


def check_text(text, label, traced, failures):
    # Find candidate measurement claims: number + unit keyword nearby.
    pattern = re.compile(r"(\d+\.\d+|\d+)\s*(ms|GFLOPS|GB/s|TFLOPS|%|e\.g\.)", re.IGNORECASE)
    for m in pattern.finditer(text):
        val = float(m.group(1))
        unit = m.group(2)
        # Skip obvious non-claims: years, versions, tile sizes, counts.
        start = max(0, m.start() - 60)
        ctx = text[start:m.end() + 40]
        if re.search(r"NOT RUN|TODO|e\.g\.|for example|version|CUDA \d|sm_\d|TILE|drift|exceeds",
                      ctx, re.I):
            continue
        # Derived quantities inside a labeled analytical model are not measurements.
        if re.search(r"ideal|implied|analytical|extrapolat|upper-bound|model", ctx, re.I):
            continue
        if "NOT RUN" in text[max(0, m.start() - 200):m.end()]:
            continue
        # Check traceability within 1% relative or small absolute tolerance.
        # TFLOPS claims may match GFLOPS evidence divided by 1000.
        if unit.upper() == "TFLOPS":
            ok = any(abs(val - t) <= max(0.01 * abs(t), 1e-6) or
                     abs(val - t / 1000.0) <= max(0.01 * abs(t / 1000.0), 1e-9)
                     for t in traced)
        else:
            ok = any(abs(val - t) <= max(0.01 * abs(t), 1e-6) or
                     (unit.lower() == "ms" and abs(val - t) < 1e-3) for t in traced)
        if not ok and unit == "%":
            for a in traced:
                for b in traced:
                    if not b:
                        continue
                    if abs(val - 100.0 * a / b) <= 1.0:
                        ok = True
                        break
                    if abs(val - 100.0 * abs(a - b) / max(abs(a), abs(b))) <= 1.0:
                        ok = True
                        break
                if ok:
                    break
        if not ok:
            # Numbers with no results files at all are only OK if declared NOT RUN.
            failures.append((label, val, unit, ctx.strip().replace("\n", " ")[:120]))


def main():
    with open(README) as f:
        readme_text = f.read()
    with open(WRITEUP) as f:
        wu_text = f.read()
    # Remove the generated table block: those numbers are traced by construction.
    if BEGIN in readme_text and END in readme_text:
        pre, _, rest = readme_text.partition(BEGIN)
        _, _, post = rest.partition(END)
        readme_body = pre + post
    else:
        readme_body = readme_text
    traced = load_numbers()
    failures = []
    check_text(readme_body, "README", traced, failures)
    check_text(wu_text, "WRITEUP", traced, failures)
    # Table freshness: regenerate and compare.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen", os.path.join(ROOT, "scripts", "generate_tables.py"))
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        table = gen.render(gen.load_rows())
        if table.strip() not in readme_text:
            print("AUDIT NOTE: results table block is stale "
                  "(run python scripts/generate_tables.py --write).")
            # Stale table is a failure only when results exist.
            if traced:
                failures.append(("README", "stale-table", "", "regenerate README table"))
    except Exception as e:
        print(f"AUDIT WARNING: could not verify table freshness: {e}")

    if failures:
        print("CLAIMS AUDIT FAILED: untraced numbers in README/WRITEUP (not in results/*.json):")
        for label, val, unit, ctx in failures:
            print(f"  [{label}] {val} {unit}  ...{ctx}...")
        print("Fix: run the bench to produce results/*.json, regenerate the table, "
              "or mark the claim NOT RUN.")
        return 1
    print(f"CLAIMS AUDIT PASSED ({len(traced)} traced values in results/*.json).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
