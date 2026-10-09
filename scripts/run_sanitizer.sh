#!/bin/bash
# Run compute-sanitizer over the correctness binary, save logs to results/sanitizer/.
# Usage: bash scripts/run_sanitizer.sh
set -u
mkdir -p results/sanitizer
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "compute-sanitizer NOT FOUND. Install the CUDA toolkit, then re-run."
  echo "Marking sanitizer evidence NOT RUN."
  exit 0
fi
BIN="${1:-build/kern_correctness}"
for TOOL in memcheck racecheck synccheck initcheck; do
  echo "[sanitizer] tool=$TOOL"
  if compute-sanitizer --tool "$TOOL" "$BIN" --seed 1234 > "results/sanitizer/${TOOL}.log" 2>&1; then
    echo "[sanitizer] $TOOL clean"
  else
    echo "[sanitizer] $TOOL FAILED - see results/sanitizer/${TOOL}.log"
  fi
done
echo "Done. Commit results/sanitizer/*.log."
