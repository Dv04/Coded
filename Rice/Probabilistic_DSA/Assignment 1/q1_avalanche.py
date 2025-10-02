import os, csv, math, sys
from pathlib import Path
import numpy as np
from sklearn.utils import murmurhash3_32

SEED_COEFFS = 580123
SEED_KEYS = 20250916

P = 1048573
A, B, C, D = 716663, 625113, 32912, 480811


def murmurhash3_32_py(data: bytes, seed: int = 0) -> int:
    return murmurhash3_32(data, seed=seed, positive=True)


def h1(x: int) -> int:
    return ((A * x + B) % P) % 1024


def h2(x: int) -> int:
    xp = x % P
    return ((A * xp * xp + B * xp + C) % P) % 1024


def h3(x: int) -> int:
    xp = x % P
    return ((A * xp * xp * xp + B * xp * xp + C * xp + D) % P) % 1024


def h4(x: int, seed: int = 137) -> int:
    b = int(x).to_bytes(4, "little", signed=False)
    return murmurhash3_32_py(b, seed=seed) % 1024


def avalanche_matrix(func, X, n_in_bits=31, n_out_bits=10) -> np.ndarray:
    M = np.zeros((n_out_bits, n_in_bits), dtype=np.float64)
    for x in X:
        y = func(int(x))
        for j in range(n_in_bits):
            y2 = func(int(x) ^ (1 << j))
            diff = y ^ y2
            for i in range(n_out_bits):
                M[i, j] += (diff >> i) & 1
    M /= float(len(X))
    return M


def run_q1():
    out_dir = Path("outputs/q1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_KEYS)
    X = rng.integers(1, 2**31, size=5000, dtype=np.uint32)

    mats = {
        "2univ_linear": avalanche_matrix(h1, X),
        "3univ_quadratic": avalanche_matrix(h2, X),
        "4univ_cubic": avalanche_matrix(h3, X),
        "murmurhash3": avalanche_matrix(lambda x: h4(x, seed=137), X),
    }

    for name, M in mats.items():
        with open(out_dir / f"{name}_avalanche.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["output_bit\\input_bit"] + [str(j) for j in range(31)])
            for i in range(10):
                w.writerow([i] + ["{:.4f}".format(M[i, j]) for j in range(31)])

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"P={P}, a={A}, b={B}, c={C}, d={D}\n")
        for name, M in mats.items():
            mean = float(np.mean(M))
            aad = float(np.mean(np.abs(M - 0.5)))
            f.write(
                f"{name}: mean={mean:.4f}, AAD_from_0.5={aad:.4f}, "
                f"min={M.min():.4f}, max={M.max():.4f}\n"
            )

    print(f"[Q1] Done. CSVs and summary in {out_dir}/")
