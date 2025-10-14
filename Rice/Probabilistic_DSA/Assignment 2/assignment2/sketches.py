from __future__ import annotations

import random
from array import array
from dataclasses import dataclass
from statistics import median
from typing import Iterable, List

from assignment2.hashing import murmurhash3_32


def _ensure_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class HashFamily:
    """Families of pairwise-independent hash/sign functions for sketch rows."""

    seeds: List[int]
    sign_seeds: List[int]
    range_size: int

    @classmethod
    def create(cls, d: int, R: int, seed: int) -> "HashFamily":
        rng = random.Random(seed)
        seeds = [rng.getrandbits(32) for _ in range(d)]
        sign_seeds = [rng.getrandbits(32) for _ in range(d)]
        return cls(seeds=seeds, sign_seeds=sign_seeds, range_size=int(R))

    def locations(self, token: str) -> Iterable[int]:
        encoded = token.encode("utf-8")
        mask = self.range_size - 1 if _ensure_power_of_two(self.range_size) else None
        for seed in self.seeds:
            h = murmurhash3_32(encoded, seed=seed)
            if mask is not None:
                yield h & mask
            else:
                yield h % self.range_size

    def signs(self, token: str) -> Iterable[int]:
        encoded = token.encode("utf-8")
        for seed in self.sign_seeds:
            h = murmurhash3_32(encoded, seed=seed)
            yield 1 if (h & 1) == 0 else -1


class CountMinSketch:
    """Standard Count-Min Sketch with point queries."""

    def __init__(self, d: int, R: int, seed: int = 0):
        self.d = int(d)
        self.R = int(R)
        self.hashes = HashFamily.create(self.d, self.R, seed)
        self.table = [array("I", [0] * self.R) for _ in range(self.d)]

    def update(self, token: str, weight: int = 1) -> None:
        for row, col in enumerate(self.hashes.locations(token)):
            self.table[row][col] += weight

    def estimate(self, token: str) -> float:
        estimates = [
            self.table[row][col]
            for row, col in enumerate(self.hashes.locations(token))
        ]
        return float(min(estimates))


class CountMedianSketch(CountMinSketch):
    """Count-Median Sketch: median-of-histogram estimator."""

    def estimate(self, token: str) -> float:
        estimates = [
            self.table[row][col]
            for row, col in enumerate(self.hashes.locations(token))
        ]
        return float(median(estimates))


class CountSketch:
    """Count-Sketch with median-of-estimates query."""

    def __init__(self, d: int, R: int, seed: int = 0):
        self.d = int(d)
        self.R = int(R)
        self.hashes = HashFamily.create(self.d, self.R, seed)
        self.table = [array("i", [0] * self.R) for _ in range(self.d)]

    def update(self, token: str, weight: int = 1) -> None:
        for row, (col, sign) in enumerate(
            zip(self.hashes.locations(token), self.hashes.signs(token))
        ):
            self.table[row][col] += int(weight) * sign

    def estimate(self, token: str) -> float:
        estimates = []
        for row, (col, sign) in enumerate(
            zip(self.hashes.locations(token), self.hashes.signs(token))
        ):
            estimates.append(self.table[row][col] * sign)
        return float(median(estimates))

