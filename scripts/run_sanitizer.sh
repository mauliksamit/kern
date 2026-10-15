#!/bin/bash
# Run compute-sanitizer over the correctness binary, save logs to results/sanitizer/.
# Usage: bash scripts/run_sanitizer.sh [path/to/kern_correctness] [build_dir]
#
# Each log starts with a provenance header (git commit + dirty flag, the
# release CUDA flags and KERN_FAST_MATH value read from the build's
# CMakeCache.txt, the binary's SHA-256, and the sanitizer version), so a log
# can be tied to the exact binary it checked.
set -u
mkdir -p results/sanitizer
SAN="$(command -v compute-sanitizer || true)"
if [ -z "$SAN" ]; then
  # Default Windows toolkit location (the sanitizer is not on PATH by default).
  for c in "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/"*/compute-sanitizer/compute-sanitizer.exe; do
    [ -x "$c" ] && SAN="$c"
  done
fi
if [ -z "$SAN" ]; then
  echo "compute-sanitizer NOT FOUND. Install the CUDA toolkit, then re-run."
  echo "Marking sanitizer evidence NOT RUN."
  exit 0
fi
BIN="${1:-build/kern_correctness}"
BUILD_DIR="${2:-$(dirname "$BIN")}"
[ -f "$BIN" ] || BIN="${BIN}.exe"

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then DIRTY=yes; else DIRTY=no; fi
CACHE="$BUILD_DIR/CMakeCache.txt"
cache_val() { grep -E "^$1:" "$CACHE" 2>/dev/null | head -1 | cut -d= -f2-; }
FAST_MATH="$(cache_val KERN_FAST_MATH)"
BUILD_TYPE="$(cache_val CMAKE_BUILD_TYPE)"
ARCH="$(cache_val KERN_CUDA_ARCH)"
# The effective release flags are set inside CMakeLists.txt (not the cache);
# recover them from the generated compile_commands.json for the kernels.
FLAGS="$(grep -o '"command": "[^"]*v6_vec_transposed.cu[^"]*"' "$BUILD_DIR/compile_commands.json" 2>/dev/null \
  | head -1 | grep -oE -- '(-O3|-lineinfo|--use_fast_math|-Xcompiler=/O2|--generate-code=[^ ]+|-arch=[^ ]+)' | tr '\n' ' ')"
SHA="$(sha256sum "$BIN" | cut -d' ' -f1)"
SANVER="$("$SAN" --version 2>&1 | grep -i version | head -1)"

for TOOL in memcheck racecheck synccheck initcheck; do
  echo "[sanitizer] tool=$TOOL"
  LOG="results/sanitizer/${TOOL}.log"
  {
    echo "# kern sanitizer log (header written by scripts/run_sanitizer.sh)"
    echo "# date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# git_commit: $COMMIT (tracked-files dirty: $DIRTY)"
    echo "# build: CMAKE_BUILD_TYPE=$BUILD_TYPE KERN_FAST_MATH=$FAST_MATH KERN_CUDA_ARCH=$ARCH"
    echo "# kernel compile flags (from compile_commands.json): $FLAGS"
    echo "# binary: $BIN sha256=$SHA"
    echo "# sanitizer: $SANVER"
    echo "# command: compute-sanitizer --tool $TOOL $BIN --seed 1234"
  } > "$LOG"
  if "$SAN" --tool "$TOOL" "$BIN" --seed 1234 >> "$LOG" 2>&1; then
    echo "[sanitizer] $TOOL clean"
  else
    echo "[sanitizer] $TOOL FAILED - see $LOG"
  fi
done
echo "Done. Commit results/sanitizer/*.log."
