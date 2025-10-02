import csv, math
from pathlib import Path
import matplotlib.pyplot as plt

N_URLS = 377_870

TAR_DIR = Path("outputs/q4")


def _read_extended_csv(path=TAR_DIR / "extended.csv"):
    rows = []
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)

        for r in rdr:
            try:
                R = int(r.get("R_bits") or r.get("Bits_R") or r["R_bits"])
                k = int(r.get("k") or r.get("Hashes_k") or r["k"])
                fpr = float(
                    r.get("empirical_fpr") or r.get("Empirical_f") or r["empirical_fpr"]
                )
                mem = int(
                    r.get("memory_bytes") or r.get("Memory_bytes") or r["memory_bytes"]
                )
                rows.append({"R_bits": R, "k": k, "fpr": fpr, "bf_bytes": mem})
            except Exception as e:
                print(f"[extended.csv] skip row {r}: {e}")
    if not rows:
        raise RuntimeError("No rows parsed from extended.csv")
    return rows


def _read_ext_mem_txt(path=TAR_DIR / "extended_memory.txt"):
    mem = {}
    if not Path(path).exists():
        return mem
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            kv = {}
            for p in parts:
                if "=" in p:
                    key, val = p.split("=", 1)
                    kv[key.strip()] = val.strip().split()[0]
            try:
                R = int(kv["R"])
                k = int(kv.get("k", "0"))
                bf_bytes = int(kv.get("bf_bytes", "0"))
                theory = int(kv.get("theory_bytes", str(R // 8)))
                pyset = int(kv.get("python_set_bytes", "0"))
                mem[R] = {
                    "k": k,
                    "bf_bytes_measured": bf_bytes,
                    "theory_bytes": theory,
                    "python_set_bytes": pyset,
                }
            except Exception as e:
                print(f"[extended_memory.txt] skip line: {line} ({e})")
    return mem


def plot_fpr_vs_memory(rows, meminfo):
    xs, ys = [], []
    for r in sorted(rows, key=lambda z: z["R_bits"]):
        R = r["R_bits"]

        x = meminfo.get(R, {}).get("theory_bytes", R // 8)
        xs.append(int(x))
        ys.append(r["fpr"])
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Memory (bytes) — theory R/8")
    plt.ylabel("Empirical FPR")
    plt.title("FPR vs Memory (policy k = floor(0.7·R/N))")
    plt.grid(True, alpha=0.3)
    TAR_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(TAR_DIR / "fpr_vs_memory.png", dpi=220)
    plt.close()
    print("[Q4] Wrote", TAR_DIR / "fpr_vs_memory.png")


def main():
    rows = _read_extended_csv()
    meminfo = _read_ext_mem_txt()
    plot_fpr_vs_memory(rows, meminfo)


if __name__ == "__main__":
    main()
