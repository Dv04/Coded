import math, sys, csv, random
from pathlib import Path
import numpy as np

SEED_BLOOM = 137


class BitArray:
    def __init__(self, nbits: int):
        self.nbits = int(nbits)
        self.bytes = bytearray((self.nbits + 7) // 8)

    def set(self, idx: int):
        b, o = (idx >> 3), (idx & 7)
        self.bytes[b] |= 1 << o

    def get(self, idx: int) -> int:
        b, o = (idx >> 3), (idx & 7)
        return (self.bytes[b] >> o) & 1

    def count(self) -> int:
        return sum(bin(x).count("1") for x in self.bytes)

    def sizeof(self) -> int:
        return sys.getsizeof(self.bytes)


def next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


def murmurhash3_32_py(data: bytes, seed: int = 0) -> int:
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    h1 = seed & 0xFFFFFFFF
    length = len(data)
    nblocks = length // 4
    for i in range(nblocks):
        k1 = int.from_bytes(data[4 * i : 4 * i + 4], "little")
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail = data[4 * nblocks :]
    k1 = 0
    if len(tail) == 3:
        k1 ^= tail[2] << 16
    if len(tail) >= 2:
        k1 ^= tail[1] << 8
    if len(tail) >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
    h1 ^= length
    h1 &= 0xFFFFFFFF
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    return h1 & 0xFFFFFFFF


def hashfunc_factory(m_bits: int, seed: int = SEED_BLOOM):
    mask = m_bits - 1

    def h(x: int, i: int):
        b = int(x).to_bytes(8, "little", signed=True)
        h1 = murmurhash3_32_py(b, seed=seed)
        h2 = murmurhash3_32_py(b, seed=(seed ^ 0x5BD1E995))
        return (h1 + i * h2) & mask

    return h


class BloomFilter:
    def __init__(self, n: int, fp_rate: float, seed: int = SEED_BLOOM):
        self.n = int(n)
        self.fp_target = float(fp_rate)
        ln2 = math.log(2.0)
        mbits = math.ceil(self.n * (-math.log(self.fp_target)) / (ln2 * ln2))
        self.m_bits = max(8, next_power_of_two(mbits))
        self.k = max(1, int(round((self.m_bits / self.n) * ln2)))
        self.seed = int(seed)
        self.bits = BitArray(self.m_bits)
        self._h = hashfunc_factory(self.m_bits, self.seed)

    def insert(self, key: int):
        for i in range(self.k):
            self.bits.set(self._h(key, i))

    def test(self, key: int) -> bool:
        return all(self.bits.get(self._h(key, i)) for i in range(self.k))

    def theoretical_fp(self) -> float:
        return (1.0 - math.exp(-self.k * self.n / self.m_bits)) ** self.k

    def memory_bytes(self) -> int:
        return self.bits.sizeof()

    def bit_count(self) -> int:
        return self.m_bits


def warmup():
    out_dir = Path("outputs/q4")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20250916)

    universe = np.arange(10_000, 100_000)
    membership = rng.choice(universe, size=10_000, replace=False)
    nonmembers = np.setdiff1d(universe, membership)
    nonmembers = rng.choice(nonmembers, size=1000, replace=False)
    trues = rng.choice(membership, size=1000, replace=False)

    targets = [0.01, 0.001, 0.0001]
    rows = []
    for f in targets:
        bf = BloomFilter(n=len(membership), fp_rate=f, seed=SEED_BLOOM)
        for x in membership:
            bf.insert(int(x))
        fps = sum(1 for x in nonmembers if bf.test(int(x)))
        fpr = fps / len(nonmembers)
        rows.append(
            [f, bf.bit_count(), bf.k, bf.theoretical_fp(), fpr, bf.memory_bytes()]
        )

    with open(out_dir / "Results.txt", "w") as f:
        f.write("TheoreticalFP | num_bits | RealFP\n")
        for r in rows:
            f.write(
                f"{r[2]}-hash on {r[1]} bits | "
                f"{r[2]} | theo={r[3]:.6g} real={r[4]:.6g}\n"
            )

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Target_f",
                "Bits_R",
                "Hashes_k",
                "Theoretical_f",
                "Empirical_f",
                "Memory_bytes",
            ]
        )
        for t, m, k, tf, rf, mem in rows:
            w.writerow([t, m, k, f"{tf:.6g}", f"{rf:.6g}", mem])

    py_set_bytes = sys.getsizeof(set(int(x) for x in membership))
    with open(out_dir / "memory.txt", "w") as f:
        f.write(
            f"Bloom memory: {rows[0][5]}..{rows[-1][5]} bytes; Python set: {py_set_bytes} bytes.\n"
        )

    print(f"[Q4] Warmup done. See outputs/q4/Results.txt and results.csv")


def extended():
    out_dir = Path("outputs/q4")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path("user-ct-test-collection-01.txt")
    if not path.exists():
        print("[Q4-extended] AOL dataset not found; skipping.")
        return

    try:
        import pandas as pd
    except Exception:
        print("[Q4-extended] pandas not installed; skipping.")
        return

    print("[Q4-extended] Loading dataset ...")
    data = pd.read_csv(path, sep="\t", quoting=3, dtype=str, on_bad_lines="skip")
    urllist = data["ClickURL"].dropna().unique()
    N = len(urllist)
    print(f"[Q4-extended] Unique URLs: N={N}")

    rng = random.Random(20250916)

    def run_for_R(R_bits):
        k = max(1, int(math.floor(0.7 * R_bits / max(N, 1))))

        class BF2(BloomFilter):
            def __init__(self, n):
                self.n = int(n)
                self.m_bits = R_bits
                self.k = k
                self.seed = SEED_BLOOM
                self.bits = BitArray(self.m_bits)
                self._h = hashfunc_factory(self.m_bits, self.seed)

        bf = BF2(n=N)
        for u in urllist:
            bf.insert(murmurhash3_32_py(u.encode("utf-8"), seed=SEED_BLOOM))
        members = [urllist[rng.randrange(N)] for _ in range(1000)]
        negs = [f"fake://{rng.getrandbits(64)}" for _ in range(1000)]
        fps = sum(
            1
            for s in negs
            if bf.test(murmurhash3_32_py(s.encode("utf-8"), seed=SEED_BLOOM))
        )
        fpr = fps / 1000.0
        mem = bf.memory_bytes()
        return (R_bits, k, fpr, mem)

    sweep = []
    for R_bits in [1 << 19, 1 << 20, 1 << 21, 1 << 22]:
        sweep.append(run_for_R(R_bits))

    with open(out_dir / "extended.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R_bits", "k", "empirical_fpr", "memory_bytes"])
        for row in sweep:
            w.writerow(row)

    print("[Q4-extended] Done. Plot FP vs memory from outputs/q4/extended.csv")

    py_set_bytes = sys.getsizeof(set(urllist))
    with open(out_dir / "extended_memory.txt", "w") as f:
        for R_bits, k, fpr, mem in sweep:
            f.write(
                f"R={R_bits} bits | k={k} | bf_bytes={mem} | theory_bytes={R_bits//8} | python_set_bytes={py_set_bytes}\n"
            )

    print("[Q4-extended] Wrote memory comparison to outputs/q4/extended_memory.txt")
