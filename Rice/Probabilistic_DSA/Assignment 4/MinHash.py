#!/usr/bin/env python3
"""
Comp 480/580 – Assignment 4
MinHash & LSH implementation plus experiment helpers.

This file provides:
  - MinHash(A, k)
  - HashTable(K, L, B, R)

and a small driver in main() that can be used to reproduce
the experiments described in the assignment.
"""

from __future__ import annotations

from typing import List, Set, Sequence, Tuple, Dict
import time
import os
import logging
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Global constants
# ---------------------------------------------------------------------------

LSH_HASH_RANGE = 1 << 20  # 2^20

# ---------------------------------------------------------------------------
#  Deterministic hash helpers
# ---------------------------------------------------------------------------


def _string_hash(s: str) -> int:
    """
    Fast deterministic polynomial hash for strings (Java-style hashCode).
    Returns a 32-bit unsigned integer.
    """
    h = 0
    for c in s:
        h = ((h * 31) + ord(c)) & 0xFFFFFFFF
    return h


def fast_minhash_code(seed: int, shingle: str) -> int:
    """
    Fast deterministic hash for MinHash: combine seed with shingle.
    Uses polynomial string hash + seed mixing for speed and reproducibility.
    Returns a 32-bit value.
    """
    h = _string_hash(shingle)
    # Mix with seed using MurmurHash-style mixing
    h ^= seed
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    return h


def stable_hash(x) -> int:
    """
    Deterministic 32-bit hash for a Python object using SHA-256.
    Used for LSH bucket mapping (not for MinHash values themselves).
    """
    data = repr(x).encode("utf-8")
    h = hashlib.sha256(data).digest()
    # take first 4 bytes as big-endian 32-bit int
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------------------
#  Utility helpers: 3-grams and Jaccard similarity
# ---------------------------------------------------------------------------


def get_kgrams(s: str, k: int = 3) -> Set[str]:
    """Return the set of character k-grams from string s (lowercased)."""
    s = s.lower()
    if len(s) < k:
        return set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def jaccard_from_sets(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def jaccard_strings(s1: str, s2: str, k: int = 3) -> float:
    """Jaccard similarity of two strings via k-gram representation."""
    A = get_kgrams(s1, k)
    B = get_kgrams(s2, k)
    return jaccard_from_sets(A, B)


# ---------------------------------------------------------------------------
#  MinHash implementation
# ---------------------------------------------------------------------------


def _minhash_from_shingles(shingles: Set[str], m: int) -> List[int]:
    """
    Compute a MinHash signature of length m for a set of shingles.

    We simulate m independent hash functions by seeding each with its index:
    h_i(shingle) = fast_minhash_code(i, shingle),
    and take the minimum hash value over all shingles for each i.
    """
    if not shingles:
        return [0] * m

    signature: List[int] = []
    for i in range(m):
        min_code = None
        for sh in shingles:
            code = fast_minhash_code(i, sh)
            if (min_code is None) or (code < min_code):
                min_code = code
        signature.append(min_code if min_code is not None else 0)
    return signature


def MinHash(A: str, k: int) -> List[int]:
    """
    Public MinHash interface required by the assignment.
    Returns a length-k signature on 3-gram shingles of A.
    """
    shingles = get_kgrams(A, k=3)
    return _minhash_from_shingles(shingles, k)


# ---------------------------------------------------------------------------
#  MinHash LSH HashTable (banding scheme: K×L MinHashes, B buckets, R range)
# ---------------------------------------------------------------------------


class HashTable:
    """
    MinHash LSH data structure using banding.

    Parameters
    ----------
    K : int
        Number of MinHash values per band (rows per table).
    L : int
        Number of bands / hash tables.
    B : int
        Number of physical buckets per table (e.g., 64).
    R : int
        Hash-code range for band keys (e.g., 2^20).
        We first hash a band key into [0, R), then fold into [0, B).
    """

    def __init__(self, K: int, L: int, B: int, R: int):
        self.K = K
        self.L = L
        self.B = B
        self.R = R

        # tables[l][b] = list of doc_ids in bucket b of table l
        self.tables: List[List[List[int]]] = [[[] for _ in range(B)] for _ in range(L)]

    def _band_bucket(self, table_id: int, signature: Sequence[int]) -> int:
        """
        Compute bucket index in table `table_id` for the given signature.

        Assumes `signature` has length exactly K*L, and table t uses
        coordinates [t*K, ..., t*K+K-1] as its band.
        """
        start = table_id * self.K
        end = start + self.K
        band_vals = signature[start:end]

        key = (table_id, tuple(band_vals))

        # 1) hash into [0, R) (space of 2^20)
        code = stable_hash(key) % self.R

        # 2) fold into one of B physical buckets
        bucket_index = code % self.B
        return bucket_index

    def insert(self, signature: Sequence[int], doc_id: int) -> None:
        """
        Insert a document id with the given MinHash signature.

        The signature length must be exactly K * L; band t uses
        coordinates [t*K, ..., t*K + K - 1].
        """
        expected_len = self.K * self.L
        if len(signature) != expected_len:
            raise ValueError(
                f"Expected signature of length {expected_len}, got {len(signature)}"
            )

        for t in range(self.L):
            b = self._band_bucket(t, signature)
            self.tables[t][b].append(doc_id)

    def lookup(self, signature: Sequence[int]) -> List[int]:
        """
        Retrieve candidate ids for the given signature from all tables.

        Returns a deduplicated list of candidate ids.
        """
        expected_len = self.K * self.L
        if len(signature) != expected_len:
            raise ValueError(
                f"Expected signature of length {expected_len}, got {len(signature)}"
            )

        candidates = set()
        for t in range(self.L):
            b = self._band_bucket(t, signature)
            for doc_id in self.tables[t][b]:
                candidates.add(doc_id)
        return list(candidates)


# ---------------------------------------------------------------------------
#  AOL URL experiments (Tasks 1–3)
# ---------------------------------------------------------------------------


def load_aol_urls(path: str) -> np.ndarray:
    """Load AOL dataset and return unique ClickURL array."""
    data = pd.read_csv(path, sep="\t")
    urllist = data["ClickURL"].dropna().unique()
    return urllist


def precompute_shingles_and_signatures(
    urllist: Sequence[str],
    sig_len: int,
) -> Tuple[List[Set[str]], List[List[int]]]:
    """
    For each URL, compute its 3-gram set and a MinHash signature of
    length sig_len. Returns (shingles_list, signatures).
    """
    shingles_list: List[Set[str]] = []
    signatures: List[List[int]] = []

    n = len(urllist)
    log_interval = max(1, n // 10)  # log every 10%

    for idx, url in enumerate(urllist):
        sh = get_kgrams(url, k=3)
        shingles_list.append(sh)
        sig = _minhash_from_shingles(sh, sig_len)
        signatures.append(sig)

        if (idx + 1) % log_interval == 0:
            logger.info(
                "Precomputing signatures: %d/%d (%.0f%%)",
                idx + 1,
                n,
                100 * (idx + 1) / n,
            )

    return shingles_list, signatures


def sample_queries(
    num_urls: int, num_queries: int = 200, seed: int = 123
) -> np.ndarray:
    """Sample query indices uniformly without replacement."""
    rng = np.random.default_rng(seed)
    return rng.choice(num_urls, size=num_queries, replace=False)


def evaluate_lsh(
    shingles_list: Sequence[Set[str]],
    full_signatures: Sequence[Sequence[int]],
    ht: HashTable,
    query_indices: Sequence[int],
    top_k: int = 10,
) -> Tuple[float, float, float]:
    """
    Evaluate LSH retrieval quality and query time.

    Returns:
        mean_jacc_all   : mean Jaccard over all retrieved candidates
        mean_jacc_topk  : mean Jaccard over top-k candidates
        time_per_query  : average lookup time in seconds
    """
    all_jaccards: List[float] = []
    topk_jaccards: List[float] = []

    K, L = ht.K, ht.L
    sig_len = K * L

    t0 = time.time()
    for q_idx in query_indices:
        q_sh = shingles_list[q_idx]
        full_sig_q = full_signatures[q_idx]
        sig_q = full_sig_q[:sig_len]

        # Lookup candidates
        cands = ht.lookup(sig_q)
        cands = [c for c in cands if c != q_idx]

        if not cands:
            continue

        sims: List[Tuple[int, float]] = []
        for c in cands:
            sim = jaccard_from_sets(q_sh, shingles_list[c])
            sims.append((c, sim))
            all_jaccards.append(sim)

        sims.sort(key=lambda x: x[1], reverse=True)
        for _, sim in sims[:top_k]:
            topk_jaccards.append(sim)

    t1 = time.time()
    total_queries = len(query_indices)
    time_per_query = (t1 - t0) / total_queries if total_queries > 0 else float("nan")

    mean_jacc_all = (
        (sum(all_jaccards) / len(all_jaccards)) if all_jaccards else float("nan")
    )
    mean_jacc_topk = (
        (sum(topk_jaccards) / len(topk_jaccards)) if topk_jaccards else float("nan")
    )

    return mean_jacc_all, mean_jacc_topk, time_per_query


def brute_force_queries(
    shingles_list: Sequence[Set[str]],
    query_indices: Sequence[int],
) -> Tuple[float, float]:
    """
    Compute brute-force Jaccard similarities between query URLs and all URLs.

    Returns:
        time_total      : total computation time
        time_per_query  : average time per query
    """
    n = len(shingles_list)
    t0 = time.time()

    for q_idx in query_indices:
        q_sh = shingles_list[q_idx]
        for i in range(n):
            if i == q_idx:
                continue
            _ = jaccard_from_sets(q_sh, shingles_list[i])

    t1 = time.time()
    time_total = t1 - t0
    time_per_query = (
        time_total / len(query_indices) if len(query_indices) > 0 else float("nan")
    )
    return time_total, time_per_query


def estimate_total_time_all_pairs(
    n_total: int,
    num_queries: int,
    measured_total_time: float,
) -> float:
    """
    Given brute-force time for 'num_queries' queries, estimate time
    for all pairwise Jaccard similarities.
    """
    pairs_computed = num_queries * (n_total - 1)
    time_per_pair = measured_total_time / pairs_computed
    total_pairs = n_total * (n_total - 1) / 2
    estimated_total_time = time_per_pair * total_pairs
    return estimated_total_time


def run_lsh_experiment_grid(
    shingles_list: Sequence[Set[str]],
    full_signatures: Sequence[Sequence[int]],
    query_indices: Sequence[int],
    Ks=(2, 3, 4, 5, 6),
    Ls=(20, 50, 100),
    B: int = 64,
    R: int = LSH_HASH_RANGE,
):
    """
    Run grid of (K, L) experiments and return list of result dicts.

    For each (K, L) we use the first K*L coordinates from the full signatures.
    B is the number of buckets per table; R is the hash-code range (2^20).
    """
    results = []
    for K in Ks:
        for L in Ls:
            logger.info("Task 3: evaluating K=%d, L=%d", K, L)
            print(f"Running LSH with K={K}, L={L}")
            ht = HashTable(K=K, L=L, B=B, R=R)
            sig_len_needed = K * L

            # Build index
            for doc_id, full_sig in enumerate(full_signatures):
                sig = full_sig[:sig_len_needed]
                ht.insert(sig, doc_id)

            mean_jacc_all, mean_jacc_top10, time_per_query = evaluate_lsh(
                shingles_list,
                full_signatures,
                ht,
                query_indices,
                top_k=10,
            )
            results.append(
                {
                    "K": K,
                    "L": L,
                    "mean_jacc_all": mean_jacc_all,
                    "mean_jacc_top10": mean_jacc_top10,
                    "time_per_query": time_per_query,
                }
            )
    logger.info("Task 3 grid search completed (%d runs)", len(results))
    return results


# ---------------------------------------------------------------------------
#  Task 4 – S-curve plots
# ---------------------------------------------------------------------------


def plot_s_curves():
    """Plot the theoretical S-curves described in Task 4 and save to PNG files."""
    logger.info("Generating Task 4 S-curve plots")
    J = np.linspace(0.0, 1.0, 200)

    # Plot 1: fix L=50, vary K
    L_fixed = 50
    plt.figure()
    for K in [1, 2, 3, 4, 5, 6, 7]:
        P = 1 - (1 - J**K) ** L_fixed
        plt.plot(J, P, label=f"K={K}")
    plt.xlabel("Jaccard similarity J_x")
    plt.ylabel("Retrieval probability P_x")
    plt.title("S-curves: L=50, varying K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("s_curve_K_varies.png")

    # Plot 2: fix K=4, vary L
    K_fixed = 4
    plt.figure()
    for L in [5, 10, 20, 50, 100, 150, 200]:
        P = 1 - (1 - J**K_fixed) ** L
        plt.plot(J, P, label=f"L={L}")
    plt.xlabel("Jaccard similarity J_x")
    plt.ylabel("Retrieval probability P_x")
    plt.title("S-curves: K=4, varying L")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("s_curve_L_varies.png")


# ---------------------------------------------------------------------------
#  Demonstration / Driver
# ---------------------------------------------------------------------------


def main():
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    logger.info("Starting Task 0 demo")

    # ---------------------------- Task 0 demo ----------------------------
    S1 = (
        "The mission statement of the WCSCC and area employers recognize the importance of good "
        "attendance on the job. Any student whose absences exceed 18 days is jeopardizing their opportunity for "
        "advanced placement as well as hindering his/her likelihood for successfully completing their program."
    )

    S2 = (
        "The WCSCC’s mission statement and surrounding employers recognize the importance of great "
        "attendance. Any student who is absent more than 18 days will loose the opportunity for successfully "
        "completing their trade program."
    )

    print("Task 0: Jaccard and MinHash estimate on S1, S2 (3-grams)")
    true_jacc = jaccard_strings(S1, S2, k=3)
    print("  True Jaccard:", true_jacc)

    m = 100
    sig1 = MinHash(S1, m)
    sig2 = MinHash(S2, m)
    equal = sum(1 for a, b in zip(sig1, sig2) if a == b)
    est_jacc = equal / m
    print("  MinHash estimated Jaccard (m=100):", est_jacc)

    # -------------------- Tasks 1–3: AOL dataset (if present) -----------
    aol_path = "user-ct-test-collection-01.txt"
    if os.path.exists(aol_path):
        print("\nAOL dataset found – running Tasks 1–3 experiments...")
        logger.info("Tasks 1–3: AOL dataset detected at %s", aol_path)
        urllist = load_aol_urls(aol_path)
        n = len(urllist)
        print("Number of unique URLs:", n)
        logger.info("Loaded %d unique URLs", n)

        # Assignment parameters: K=2, L=50, B=64, R=2^20.
        K_base, L_base = 2, 50
        B_base = 64
        R_base = LSH_HASH_RANGE  # 2^20

        # For the grid K∈{2,...,6}, L∈{20,50,100} we need up to 6*100 = 600 coords.
        Ks_grid = (2, 3, 4, 5, 6)
        Ls_grid = (20, 50, 100)
        sig_len_max = max(Ks_grid) * max(Ls_grid)  # 6 * 100 = 600

        logger.info("Precomputing shingles and signatures (length %d)", sig_len_max)
        shingles_list, full_signatures = precompute_shingles_and_signatures(
            urllist, sig_len=sig_len_max
        )

        # Sample queries
        q_indices = sample_queries(n, num_queries=200, seed=42)

        # Build baseline LSH index for K=2, L=50 using first 100 coords
        logger.info(
            "Building baseline LSH index (K=%d, L=%d, B=%d, R=%d)",
            K_base,
            L_base,
            B_base,
            R_base,
        )
        ht_base = HashTable(K=K_base, L=L_base, B=B_base, R=R_base)
        sig_len_base = K_base * L_base  # 100

        for doc_id, full_sig in enumerate(full_signatures):
            sig = full_sig[:sig_len_base]
            ht_base.insert(sig, doc_id)

        logger.info("Running LSH evaluation for %d sampled queries", len(q_indices))
        mean_jacc_all, mean_jacc_top10, time_per_query = evaluate_lsh(
            shingles_list,
            full_signatures,
            ht_base,
            q_indices,
            top_k=10,
        )

        print("\nTask 1:")
        print("  Mean Jaccard over all retrieved candidates:", mean_jacc_all)
        print("  Mean Jaccard over top-10 candidates:", mean_jacc_top10)
        print("  Average LSH query time (seconds/query):", time_per_query)

        # -------------------- Task 2: brute-force baseline --------------
        bf_total_time, bf_time_per_query = brute_force_queries(shingles_list, q_indices)
        est_total_time_all_pairs = estimate_total_time_all_pairs(
            n, len(q_indices), bf_total_time
        )

        logger.info(
            "Completed brute-force baseline: total %.2fs, avg %.4fs/query",
            bf_total_time,
            bf_time_per_query,
        )

        print("\nTask 2:")
        print("  Brute-force total time for 200 queries:", bf_total_time)
        print("  Brute-force time per query:", bf_time_per_query)
        print("  Estimated time for ALL pairwise Jaccards:", est_total_time_all_pairs)

        # -------------------- Task 3: K,L grid search -------------------
        grid_results = run_lsh_experiment_grid(
            shingles_list,
            full_signatures,
            q_indices,
            Ks=Ks_grid,
            Ls=Ls_grid,
            B=B_base,
            R=R_base,
        )

        print("\nTask 3 results (per (K, L) combination):")
        for r in grid_results:
            print(
                f"  K={r['K']}, L={r['L']}, "
                f"mean_jacc_all={r['mean_jacc_all']:.4f}, "
                f"mean_jacc_top10={r['mean_jacc_top10']:.4f}, "
                f"time_per_query={r['time_per_query']:.6f} sec"
            )
    else:
        print("\nAOL dataset file 'user-ct-test-collection-01.txt' not found.")
        print(
            "Skipping Tasks 1–3 (LSH experiments). Place the file next to this script to run them."
        )
        logger.warning("Tasks 1–3 skipped because %s is missing", aol_path)

    # -------------------------- Task 4: S-curves ------------------------
    print("\nGenerating S-curve plots (Task 4)...")
    plot_s_curves()
    print("S-curve plots saved as 's_curve_K_varies.png' and 's_curve_L_varies.png'.")


if __name__ == "__main__":
    main()
