from __future__ import annotations

"""
Self-contained Assignment 2 script
- Implements hashing (MurmurHash3 x86 32-bit)
- Implements Count-Min, Count-Median, Count-Sketch with pairwise-independent hash families
- Processes AOL dataset, computes errors on Frequent-100 / Random-100 / Infrequent-100
- Tracks Top-500 online estimates and measures intersection with true Top-100
- Generates plots and LaTeX tables, and summary.json under outputs/a2

This file consolidates functionality from hashing.py, sketches.py and main_a2.py
so it can be run standalone (no external mmh3 dependency required).
"""

import argparse
import json
import logging
import math
import random
import re
import sys
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple

# ==========================
# Hashing (MurmurHash3 32)
# ==========================

def murmurhash3_32(data: bytes, seed: int = 0) -> int:
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


def _ensure_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


# ==========================
# Sketch hash families
# ==========================

@dataclass(frozen=True)
class HashFamily:
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


# ==========================
# Sketches
# ==========================

class CountMinSketch:
    def __init__(self, d: int, R: int, seed: int = 0):
        self.d = int(d)
        self.R = int(R)
        self.hashes = HashFamily.create(self.d, self.R, seed)
        self.table = [array("I", [0] * self.R) for _ in range(self.d)]

    def update(self, token: str, weight: int = 1) -> None:
        for row, col in enumerate(self.hashes.locations(token)):
            self.table[row][col] += int(weight)

    def estimate(self, token: str) -> float:
        estimates = [
            self.table[row][col]
            for row, col in enumerate(self.hashes.locations(token))
        ]
        return float(min(estimates))


class CountMedianSketch(CountMinSketch):
    def estimate(self, token: str) -> float:
        estimates = [
            self.table[row][col]
            for row, col in enumerate(self.hashes.locations(token))
        ]
        return float(median(estimates))


class CountSketch:
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


# ==========================
# Tokenisation & dataset IO
# ==========================

DEFAULT_DATASET = "user-ct-test-collection-01.txt"
R_VALUES = [2**10, 2**14, 2**18]
D_ROWS = 5
TOP_K = 500
RANDOM_SEED = 20251013


def tokenize_query(text: str) -> Iterable[str]:
    for token in re.findall(r"[A-Za-z0-9']+", text.lower()):
        if token:
            yield token


def iter_tokens(dataset: Path, limit_rows: int | None = None) -> Iterable[str]:
    with dataset.open("r", encoding="utf-8", errors="ignore") as fh:
        header = next(fh, None)
        if header is None:
            return
        for row_idx, line in enumerate(fh, start=1):
            if limit_rows and row_idx > limit_rows:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            for token in tokenize_query(parts[1]):
                yield token


# ==========================
# Top-K tracker
# ==========================

class TopKTracker:
    def __init__(self, k: int):
        import heapq

        self.k = int(k)
        self.heapq = heapq
        self.heap: List[Tuple[float, str]] = []
        self.estimates: Dict[str, float] = {}

    def update(self, token: str, value: float) -> None:
        value = float(max(value, 0.0))
        prev = self.estimates.get(token)
        if prev is None:
            if len(self.heap) < self.k:
                self.estimates[token] = value
                self.heapq.heappush(self.heap, (value, token))
            elif value > self.heap[0][0]:
                removed_value, removed_token = self.heapq.heapreplace(
                    self.heap, (value, token)
                )
                if removed_token in self.estimates:
                    del self.estimates[removed_token]
                self.estimates[token] = value
        elif value > prev:
            self.estimates[token] = value
            self.heapq.heappush(self.heap, (value, token))

    def _trim(self) -> None:
        while self.heap:
            value, token = self.heap[0]
            if self.estimates.get(token) == value:
                break
            self.heapq.heappop(self.heap)

    def top_items(self, size: int | None = None) -> List[Tuple[str, float]]:
        self._trim()
        size = min(size or self.k, len(self.estimates))
        return sorted(self.estimates.items(), key=lambda kv: kv[1], reverse=True)[:size]

    def as_set(self) -> set[str]:
        return {token for token, _ in self.top_items()}


# ==========================
# Metrics, plotting and tables
# ==========================

def approx_counter_size(counter: Counter[str]) -> int:
    size = sys.getsizeof(counter)
    for key, value in counter.items():
        size += sys.getsizeof(key)
        size += sys.getsizeof(value)
    return size


def compute_errors(
    sketch, items: List[Tuple[str, int]]
) -> Tuple[List[float], Dict[str, float]]:
    errs: List[float] = []
    for token, truth in items:
        truth = max(int(truth), 1)
        estimate = sketch.estimate(token)
        errs.append(abs(estimate - truth) / truth)
    if errs:
        stats = {
            "mean": float(mean(errs)),
            "median": float(median(errs)),
            "max": float(max(errs)),
        }
    else:
        stats = {"mean": math.nan, "median": math.nan, "max": math.nan}
    return errs, stats


def plot_error_curves(
    errors: Dict[str, List[float]],
    tokens: List[Tuple[str, int]],
    title: str,
    output_path: Path,
    plt,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = list(range(len(tokens)))
    for label, series in errors.items():
        ax.plot(xs, series, marker="o", markersize=2, linewidth=1, label=label)
    positive = [value for series in errors.values() for value in series if value > 0]
    if positive:
        ax.set_yscale("log")
        min_positive = min(positive)
        ax.set_ylim(bottom=max(min_positive / 10.0, 1e-4))
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlabel("Tokens (sorted by frequency)")
    ax.set_ylabel("Relative error")
    ax.set_title(title)
    if len(tokens) <= 100:
        labels = [tok if len(tok) <= 20 else f"{tok[:19]}…" for tok, _ in tokens]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_topk_intersection(
    intersections: Dict[str, List[int]], output_path: Path, plt
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = R_VALUES
    for label, series in intersections.items():
        ax.plot(xs, series, marker="o", linewidth=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Range size R")
    ax.set_ylabel("|Top-500 ∩ True Top-100|")
    ax.set_title("Intersection with True Top-100")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


_MATPLOTLIB = None


def _format_int(value: int) -> str:
    return f"{int(value):,}".replace(",", "\\,")


def _format_value(value) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf"
        formatted = f"{value:.3f}"
        formatted = formatted.rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    return str(value)


def write_latex_tables(summary: Dict, out_dir: Path) -> None:
    error_summary = summary.get("error_summary") or {}
    if error_summary:
        r_keys = sorted(error_summary.keys(), key=lambda k: int(k))
        categories = ["Frequent-100", "Random-100", "Infrequent-100"]
        sketches = ["Count-Min", "Count-Median", "Count-Sketch"]
        lines = [
            "\\begin{tabular}{lllccc}",
            "\\toprule",
                "Category & Sketch & $R$ & Mean & Median & Max \\\\",
            "\\midrule",
        ]
        for c_idx, category in enumerate(categories):
            for s_idx, sketch in enumerate(sketches):
                for r_idx, r_key in enumerate(r_keys):
                    stats = (
                        error_summary.get(r_key, {}).get(sketch, {}).get(category, {})
                    )
                    mean_val = _format_value(stats.get("mean", float("nan")))
                    median_val = _format_value(stats.get("median", float("nan")))
                    max_val = _format_value(stats.get("max", float("nan")))
                    cat_label = category if s_idx == 0 and r_idx == 0 else ""
                    sketch_label = sketch if r_idx == 0 else ""
                    r_label = f"$2^{{{int(math.log2(int(r_key)))}}}$"
                    lines.append(
                        f"{cat_label} & {sketch_label} & {r_label} & {mean_val} & {median_val} & {max_val} \\\")
            if c_idx < len(categories) - 1:
                lines.append("\\midrule")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        (out_dir / "error_table.tex").write_text("\n".join(lines), encoding="utf-8")

    intersections = summary.get("intersection_top100") or {}
    if intersections:
        r_values = summary.get("parameters", {}).get("R_values", [])
        if not r_values:
            r_values = sorted({int(k) for k in next(iter(intersections.values()), {})})
        header_cols = [f"$R=2^{{{int(math.log2(r))}}}$" for r in r_values]
        lines = [
            f"\\begin{{tabular}}{{l{'c'*len(r_values)}}}",
            "\\toprule",
                "Sketch & " + " & ".join(header_cols) + " \\\\",
            "\\midrule",
        ]
        for sketch, values in intersections.items():
            row_entries = []
            for r in r_values:
                key = str(r)
                row_entries.append(str(values.get(key, "--")))
            lines.append(f"{sketch} & " + " & ".join(row_entries) + " \\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        (out_dir / "intersection_table.tex").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        col_span = len(r_values) + 1
        median_lines = [
            f"\\begin{{tabular}}{{l{'c'*len(r_values)}}}",
            "\\toprule",
            "Sketch & " + " & ".join(header_cols) + " \\\",
            "\\midrule",
            f"\\multicolumn{{{col_span}}}{{c}}{{Frequent-100 median relative error}} \\",
            "\\midrule",
        ]
        for sketch in intersections:
            row = []
            for r in r_values:
                stats = (
                    error_summary.get(str(r), {})
                    .get(sketch, {})
                    .get("Frequent-100", {})
                )
                row.append(_format_value(stats.get("median", float("nan"))))
            median_lines.append(f"{sketch} & " + " & ".join(row) + " \\")
        median_lines.append("\\addlinespace")
        median_lines.append("\\midrule")
        median_lines.append(
            f"\\multicolumn{{{col_span}}}{{c}}{{Random-100 median relative error}} \\")
        median_lines.append("\\midrule")
        for sketch in intersections:
            row = []
            for r in r_values:
                stats = (
                    error_summary.get(str(r), {}).get(sketch, {}).get("Random-100", {})
                )
                row.append(_format_value(stats.get("median", float("nan"))))
            median_lines.append(f"{sketch} & " + " & ".join(row) + " \\")
        median_lines.append("\\addlinespace")
        median_lines.append("\\midrule")
        median_lines.append(
            f"\\multicolumn{{{col_span}}}{{c}}{{Infrequent-100 median relative error}} \\")
        median_lines.append("\\midrule")
        for sketch in intersections:
            row = []
            for r in r_values:
                stats = (
                    error_summary.get(str(r), {})
                    .get(sketch, {})
                    .get("Infrequent-100", {})
                )
                row.append(_format_value(stats.get("median", float("nan"))))
            median_lines.append(f"{sketch} & " + " & ".join(row) + " \\")
        median_lines.append("\\bottomrule")
        median_lines.append("\\end{tabular}")
        (out_dir / "median_table.tex").write_text(
            "\n".join(median_lines), encoding="utf-8"
        )

    dataset_str = str(summary.get("parameters", {}).get("dataset", ""))
    dataset_tex = dataset_str.replace("_", "\\_")
    limit_rows = summary.get("parameters", {}).get("limit_rows", 0)
    if limit_rows in (0, None):
        limit_text = "All rows"
    else:
        limit_text = _format_int(limit_rows)
    dict_bytes = summary.get("dictionary_bytes_estimate", 0)
    dict_mib = dict_bytes / (1024**2) if dict_bytes else 0
    run_lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
            "Metric & Value \\\\",
        "\\midrule",
        f"Processed tokens & {_format_int(summary.get('total_tokens', 0))} \\",
        f"Unique tokens & {_format_int(summary.get('unique_tokens', 0))} \\",
        f"Dictionary size (MiB) & {_format_value(dict_mib)} \\",
        f"Row budget & {limit_text} \\",
        f"Dataset flag & \\texttt{{--dataset~{dataset_tex}}} \\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (out_dir / "run_summary.tex").write_text("\n".join(run_lines), encoding="utf-8")


def ensure_matplotlib():
    global _MATPLOTLIB
    if _MATPLOTLIB is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _MATPLOTLIB = plt
    return _MATPLOTLIB


# ==========================
# Main
# ==========================

def main() -> None:
    parser = argparse.ArgumentParser(description="COMP 480/580 Assignment 2 runner (single-file)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(DEFAULT_DATASET),
        help="Path to AOL dataset (tab separated).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on rows processed (0 means all rows).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/a2"),
        help="Directory to store plots and metrics.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating PNG plots (useful on headless systems).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200_000,
        help="Log progress every N tokens processed.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("assignment2")
    logger.info("Starting Assignment 2 run with dataset=%s", args.dataset)
    if args.limit > 0:
        logger.info("Row limit set to %d", args.limit)

    rng = random.Random(RANDOM_SEED)

    dataset = args.dataset
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    logger.debug("Dataset located at %s", dataset.resolve())

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Outputs will be written to %s", out_dir.resolve())
    plt_obj = None if args.skip_plots else ensure_matplotlib()
    if plt_obj is None:
        logger.info("Plot generation disabled (skip_plots=True).")
    else:
        logger.debug("Matplotlib backend initialised.")

    sketches: Dict[int, Dict[str, object]] = {}
    trackers: Dict[int, Dict[str, TopKTracker]] = {}
    for R in R_VALUES:
        sketches[R] = {
            "Count-Min": CountMinSketch(D_ROWS, R, seed=RANDOM_SEED + R),
            "Count-Median": CountMedianSketch(D_ROWS, R, seed=RANDOM_SEED + 7 * R),
            "Count-Sketch": CountSketch(D_ROWS, R, seed=RANDOM_SEED + 13 * R),
        }
        trackers[R] = {name: TopKTracker(TOP_K) for name in sketches[R]}
        logger.debug("Initialised sketches for R=%d", R)

    counter: Counter[str] = Counter()
    total_tokens = 0
    log_interval = max(args.log_interval, 1)
    logger.info("Beginning token processing with log interval=%d", log_interval)

    for token in iter_tokens(dataset, args.limit if args.limit > 0 else None):
        counter[token] += 1
        total_tokens += 1
        for R in R_VALUES:
            for name, sketch in sketches[R].items():
                sketch.update(token, 1)
                estimate = sketch.estimate(token)
                trackers[R][name].update(token, estimate)
        if total_tokens % log_interval == 0:
            logger.info("Processed %d tokens so far...", total_tokens)

    unique_tokens = len(counter)
    logger.info(
        "Completed token processing: total=%d, unique=%d",
        total_tokens,
        unique_tokens,
    )

    freq_items = counter.most_common(100)
    infreq_items = sorted(counter.items(), key=lambda kv: (kv[1], kv[0]))[:100]
    rand_items = (
        rng.sample(list(counter.items()), k=min(100, unique_tokens))
        if unique_tokens
        else []
    )
    if rand_items:
        rand_items = sorted(rand_items, key=lambda kv: kv[1], reverse=True)

    categories: Dict[str, List[Tuple[str, int]]] = {
        "Frequent-100": freq_items,
        "Infrequent-100": infreq_items,
        "Random-100": rand_items,
    }

    error_summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    dictionary_bytes = approx_counter_size(counter)

    for R in R_VALUES:
        error_summary[str(R)] = {}
        for name in sketches[R]:
            error_summary[str(R)][name] = {}
        for label, items in categories.items():
            errors_for_plot: Dict[str, List[float]] = {}
            for name, sketch in sketches[R].items():
                errors, stats = compute_errors(sketch, items)
                error_summary[str(R)][name][label] = stats
                errors_for_plot[name] = errors
            plot_error_curves(
                errors_for_plot,
                items,
                f"{label} errors @ R={R}",
                out_dir / f"errors_R{R}_{label.replace('-', '_')}.png",
                plt_obj,
            )
            logger.debug("Computed error stats for category=%s at R=%d", label, R)

    intersections: Dict[str, List[int]] = {
        "Count-Min": [],
        "Count-Median": [],
        "Count-Sketch": [],
    }
    true_top100 = {token for token, _ in freq_items[:100]}
    for R in R_VALUES:
        for name in intersections:
            approx_set = trackers[R][name].as_set()
            intersections[name].append(len(approx_set & true_top100))

    plot_topk_intersection(intersections, out_dir / "top500_intersection.png", plt_obj)
    logger.info("Computed top-k intersections and generated plots.")

    summary = {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "dictionary_bytes_estimate": dictionary_bytes,
        "error_summary": error_summary,
        "intersection_top100": {
            name: dict(zip([str(R) for R in R_VALUES], sizes))
            for name, sizes in intersections.items()
        },
        "parameters": {
            "rows_d": D_ROWS,
            "R_values": R_VALUES,
            "top_k": TOP_K,
            "seed": RANDOM_SEED,
            "dataset": str(dataset),
            "limit_rows": args.limit,
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary metrics written to %s", out_dir / "summary.json")
    write_latex_tables(summary, out_dir)

    print(
        f"[A2] Processed {total_tokens:,} tokens across {unique_tokens:,} unique terms."
    )
    print(f"[A2] Approximate dictionary size: {dictionary_bytes / (1024**2):.2f} MiB.")
    if plt_obj is not None:
        print(f"[A2] Plots written to {out_dir}/")
    logger.info("Assignment 2 run completed.")


if __name__ == "__main__":
    main()
