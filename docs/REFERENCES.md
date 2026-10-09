# References

Every idea adapted from these is noted in a code comment + here.
No code was copied from these references; all kernels were written from scratch.

1. CUDA C++ Programming Guide — memory coalescing, shared memory, occupancy.
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/
   Used in: `kernels/v1_coalesced.cu`, `kernels/v2_tiled.cu` (idea attribution in file headers).
2. CUDA C++ Best Practices Guide — vectorized access, `__launch_bounds__`, `-lineinfo`.
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
   Used in: `kernels/v3_tiled_vectorized.cu`, `CMakeLists.txt` release flags.
3. CUDA sample `matrixMul` / "Tiled Matrix Multiplication" (UIUC / NVIDIA samples) —
   classic shared-memory tiling pattern.
   Used in: `kernels/v2_tiled.cu` (pattern attribution; implementation original).
4. cuBLAS documentation (`cublasSgemm`, row-major emulation via transposed operands).
   https://docs.nvidia.com/cuda/cublas/
   Used in: `src/bench.cu` `time_cublas()`.
5. Nsight Compute docs — `--query-metrics`, `--list-sections` (metric names must be
   confirmed per installed version, never assumed).
   https://docs.nvidia.com/nsight-compute/
   Used in: `scripts/run_ncu.sh`, `docs/WRITEUP.md` profiling plan.
6. Blackwell / RTX 50-series compatibility — CUDA >= 12.8 required for sm_120.
   https://developer.nvidia.com/cuda-toolkit-archive
   Used in: `docs/ENVIRONMENT.md` arch inference, `tools/detect_arch.py`.
7. CUDA C++ Programming Guide — `__syncthreads()` barrier uniformity
   ("__syncthreads() is allowed in conditional code only if the conditional
   evaluates identically across the entire thread block, otherwise ... execution
   is undefined" / independent thread scheduling notes).
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/
   Used in: `kernels/v3_tiled_vectorized.cu` (2026-10-09 fix: fast-path decision
   hoisted from per-thread row/col alignment to block-uniform tile-base
   alignment after a divergent-barrier hang; see `LOG.md`).
