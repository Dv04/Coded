# Dev Sanghvi (ds221)

"""Utility to convert sweep CSVs into LaTeX tables (report/*.tex).
Runs standalone: python src/gen_report_tables.py"""

import csv
import argparse
from pathlib import Path


def fmt_acc(x):
    try:
        return f"{float(x) * 100:.2f}\\%"
    except Exception:
        return "--"


def load_cnn_rows(csv_path: Path):
    rows = []
    if not csv_path or not csv_path.exists():
        return rows
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            opt = row.get("optimizer", "")
            lr = row.get("lr", "")
            bs = row.get("batch_size", "")
            act = row.get("activation", "")
            mom = row.get("momentum", "")
            te = row.get("test_acc", "")
            rows.append((opt, lr, bs, act, mom, te))
    try:
        rows.sort(key=lambda t: float(t[-1]), reverse=True)
    except Exception:
        pass
    return rows


def load_mnist_rows(csv_path: Path):
    rows = []
    if not csv_path or not csv_path.exists():
        return rows
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("rnn_type", "").upper()
            hs = row.get("hidden_size", "")
            layers = row.get("num_layers", "")
            bi = "Yes" if row.get("bidirectional") in ("True", True, "yes", "1", 1) else "No"
            te = row.get("test_acc", "")
            rows.append((model, hs, layers, bi, te))
    try:
        rows.sort(key=lambda t: float(t[-1]), reverse=True)
    except Exception:
        pass
    return rows


def write_rows(tex_path: Path, rows, fallback_multicol: str):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w") as f:
        if rows:
            for r in rows:
                f.write(" & ".join(str(x) if x is not None else "" for x in r[:-1]))
                f.write(f" & {fmt_acc(r[-1])} \\\n")
        else:
            f.write(f"{fallback_multicol}\\\\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_csv", default=None, type=str)
    ap.add_argument("--mnist_csv", default=None, type=str)
    ap.add_argument("--out_dir", default=None, type=str)
    ap.add_argument("--cnn_out", default="cnn_hp_table.tex", type=str)
    ap.add_argument("--mnist_out", default="mnist_hp_table.tex", type=str)
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    default_cnn = repo_root / "outputs" / "csv" / "search_results_cnn.csv"
    default_mnist = repo_root / "outputs" / "csv" / "search_results_mnist.csv"
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "report")

    cnn_csv = Path(args.cnn_csv) if args.cnn_csv else default_cnn
    mn_csv = Path(args.mnist_csv) if args.mnist_csv else default_mnist

    cnn_rows = load_cnn_rows(cnn_csv)
    mn_rows = load_mnist_rows(mn_csv)

    write_rows(out_dir / args.cnn_out, cnn_rows, "\\multicolumn{6}{c}{(No CNN sweep CSV found)}")
    write_rows(out_dir / args.mnist_out, mn_rows, "\\multicolumn{5}{c}{(No MNIST sweep CSV found)}")

    print(f"[INFO] repo_root: {repo_root}")
    print(f"[INFO] cnn_csv:   {cnn_csv} -> {out_dir / args.cnn_out}")
    print(f"[INFO] mnist_csv: {mn_csv} -> {out_dir / args.mnist_out}")
    if cnn_rows:
        best = cnn_rows[0]
        print(
            f"[CNN] Best: optimizer={best[0]}, lr={best[1]}, batch={best[2]}, act={best[3]}, momentum={best[4]}, test_acc={fmt_acc(best[5])}"
        )
    if mn_rows:
        best = mn_rows[0]
        print(
            f"[MNIST] Best: model={best[0]}, hidden={best[1]}, layers={best[2]}, bi={best[3]}, test_acc={fmt_acc(best[4])}"
        )


if __name__ == "__main__":
    main()
