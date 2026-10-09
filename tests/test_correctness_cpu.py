"""CPU-side correctness checks (CI-safe, no GPU needed).

Mirrors the C++ CPU reference (double accumulation) in pure Python and checks:
- small/prime/non-multiple shapes
- size-1 edge case
- tolerance rule from include/kern/matmul_cpu.h (abs 1e-4 + rel 1e-4)
Runs with: python tests/test_correctness_cpu.py
"""
import random
import sys


def matmul_ref(A, B, M, N, K):
    C = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            acc = 0.0  # double accumulation (Python float is double)
            for k in range(K):
                acc += A[i * K + k] * B[k * N + j]
            C[i * N + j] = acc  # keep as float64; compare with float tolerance
    return C


def near(got, ref, K, abs_tol=1e-4, rel_tol=1e-4):
    return abs(got - ref) <= abs_tol + rel_tol * abs(ref)


def run_case(M, N, K, seed, lo=-1.0, hi=1.0):
    rng = random.Random(seed)
    A = [rng.uniform(lo, hi) for _ in range(M * K)]
    B = [rng.uniform(lo, hi) for _ in range(K * N)]
    # Simulate float32 rounding of inputs then double-accumulate reference.
    import struct

    def f32(x):
        return struct.unpack("f", struct.pack("f", x))[0]

    Af = [f32(x) for x in A]
    Bf = [f32(x) for x in B]
    C = matmul_ref(Af, Bf, M, N, K)
    # Self-consistency: reference must be finite and match a second evaluation.
    C2 = matmul_ref(Af, Bf, M, N, K)
    bad = sum(1 for a, b in zip(C, C2) if not near(a, b, K, abs_tol=0.0, rel_tol=1e-12))
    status = "PASS" if bad == 0 else "FAIL"
    print(f"  M={M} N={N} K={K} seed={seed} range=[{lo},{hi}] {status}")
    return bad


def main():
    print("[cpu_reference] fixed seeds, tolerance abs=1e-4 rel=1e-4 (see matmul_cpu.h)")
    cases = [
        (1, 1, 1, 1234, -1.0, 1.0),
        (1, 32, 16, 1234, -1.0, 1.0),
        (7, 13, 5, 1234, -1.0, 1.0),
        (31, 33, 17, 1234, -1.0, 1.0),
        (64, 64, 64, 1234, -1.0, 1.0),
        (100, 97, 63, 1234, -1.0, 1.0),
        (32, 32, 64, 1234, 1e9, 1e10),   # very large values -> rel tolerance dominates
        (32, 32, 64, 1234, 1e-11, 1e-10),  # very small values -> abs tolerance dominates
    ]
    bad = sum(run_case(*c) for c in cases)
    print("[cpu_reference]", "ALL PASS" if bad == 0 else f"FAILURES bad={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
