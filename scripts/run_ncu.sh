#!/bin/bash
# Nsight Compute profiling helper.
# Never assume metric/section names: confirm with the installed version first:
#   ncu --query-metrics | grep -i -E "bank|dram|sm__throughput"
#   ncu --list-sections
# Then profile each major step and save the summary next to results/.
# Usage: bash scripts/run_ncu.sh [M N K]  (defaults 1024 1024 1024)
set -u
M="${1:-1024}"; N="${2:-1024}"; K="${3:-1024}"
mkdir -p results/ncu
if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu NOT FOUND on this machine. Mark profiler evidence NOT RUN;"
  echo "rely on cudaEvent timing + clearly labeled analytical reasoning."
  exit 0
fi
echo "== installed sections =="; ncu --list-sections | head -30
echo "== sample metrics =="; ncu --query-metrics | grep -i -E "sm__throughput|dram__throughput|shared" | head -20
BIN="${BIN:-build/kern_bench}"
for KERNEL in v0 v1 v2 v3; do
  echo "[ncu] profiling $KERNEL M=$M N=$N K=$K"
  ncu --set detailed --section SpeedOfLight --section MemoryWorkloadAnalysis \
      --export "results/ncu/${KERNEL}_M${M}_N${N}_K${K}" \
      "$BIN" --M "$M" --N "$N" --K "$K" \
      --out "results/bench_tmp_ncu.json" > "results/ncu/${KERNEL}_M${M}_N${N}_K${K}.log" 2>&1 || true
done
echo "Saved results/ncu/*.log + *.ncu-rep. Tie each WRITEUP claim to a metric value."
