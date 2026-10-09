#!/usr/bin/env python3
"""Detect GPU arch for -DKERN_CUDA_ARCH (run on a CUDA machine).

Uses nvidia-smi + nvcc/deviceQuery when available. RTX 50-series (Blackwell)
needs CUDA >= 12.8 and -arch sm_120.
"""
import shutil
import subprocess

print("== nvidia-smi ==")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                          "--format=csv"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    print("nvidia-smi not found")

if shutil.which("nvcc"):
    print(subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout)
else:
    print("nvcc NOT found: install CUDA toolkit 12.8+ (for RTX 50-series sm_120).")
    print("Then configure with: cmake -S . -B build -DKERN_CUDA_ARCH=sm_120")

KNOWN = {
    "RTX 5070": "sm_120", "RTX 5080": "sm_120", "RTX 5090": "sm_120",
    "RTX 4090": "sm_90", "RTX 4080": "sm_90", "A100": "sm_80", "V100": "sm_70",
}
print("Known mapping (verify with CUDA docs for your GPU):", KNOWN)
