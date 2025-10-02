import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def _load_matrix(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None) 
        for r in reader:
            if not r or all((c or "").strip() == "" for c in r):
                continue
            vals = []
            for cell in r[1:] if len(r) > 1 else r:
                cell = (cell or "").strip()
                if cell:
                    vals.append(float(cell))
            if vals:
                rows.append(vals)
    M = np.array(rows, dtype=float)
    if M.ndim != 2 or M.size == 0:
        raise ValueError(f"Invalid/empty matrix from {csv_path} (shape={M.shape})")
    return M


def _diverging_white_center():
    return LinearSegmentedColormap.from_list(
        "center_dark_div",
        [
            (0.0, "#c6dbef"),  # light blue
            (0.25, "#6baed6"),
            (0.50, "#000000"),  # DARK at 0.5
            (0.75, "#fcae91"),
            (1.0, "#fee5d9"),  # light red
        ],
    )


def _auto_limits_around_half(M, min_width=0.02, q=0.99):
    lo = np.quantile(M, 1 - q)
    hi = np.quantile(M, q)
    w = max(min_width, max(0.5 - lo, hi - 0.5))
    vmin = max(0.0, 0.5 - w)
    vmax = min(1.0, 0.5 + w)
    return vmin, vmax


def save_prob_heatmap(csv_path, out_png, title):
    M = _load_matrix(csv_path)
    vmin, vmax = _auto_limits_around_half(M, min_width=0.02, q=0.995)
    plt.figure(figsize=(8, 3.2))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)
    plt.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        norm=norm,
        cmap=_diverging_white_center(),
    )
    cb = plt.colorbar()
    cb.set_label("P(output bit flips | input bit flips)")

    ticks = sorted(set([vmin, 0.5, vmax]))
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.3f}" for t in ticks])

    plt.xlabel("input bit j (0..30)")
    plt.ylabel("output bit i (0..9)")
    plt.title(title)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_deviation_heatmap(csv_path, out_png, title):
    M = _load_matrix(csv_path)
    D = np.abs(M - 0.5)
    vmax = max(0.01, np.quantile(D, 0.995))
    plt.figure(figsize=(8, 3.2))
    plt.imshow(
        D,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
        cmap=plt.get_cmap("magma"),
    )
    cb = plt.colorbar()
    cb.set_label(r"$|P - 0.5|$")
    plt.xlabel("input bit j (0..30)")
    plt.ylabel("output bit i (0..9)")
    plt.title(title + " (deviation from 0.5)")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def make_all():
    out_dir = Path("outputs/q1")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("outputs/q1/2univ_linear_avalanche.csv", "2-universal (linear)"),
        ("outputs/q1/3univ_quadratic_avalanche.csv", "3-universal (quadratic)"),
        ("outputs/q1/4univ_cubic_avalanche.csv", "4-universal (cubic)"),
        ("outputs/q1/murmurhash3_avalanche.csv", "MurmurHash3"),
    ]
    for src, title in tasks:
        p = Path(src)
        if not p.exists():
            print(f"[Q1] Missing {src}, skipping.")
            continue
        save_prob_heatmap(
            src, p.with_name(p.stem.replace("_avalanche", "") + "_heatmap.png"), title
        )
        save_deviation_heatmap(
            src,
            p.with_name(p.stem.replace("_avalanche", "") + "_dev_heatmap.png"),
            title,
        )


if __name__ == "__main__":
    make_all()
