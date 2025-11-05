# Dev Sanghvi (ds221)

"""Hyper-parameter sweep harness; runs training scripts via subprocess and logs CSVs.
Runs standalone: python src/search.py --task cnn --budget_epochs 15"""

import os
import argparse
import subprocess
import sys
import time
from typing import Dict, Any, List

import pandas as pd


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dur = time.time() - start
    print(proc.stdout)
    return proc.returncode, dur, proc.stdout


def parse_metrics(output: str):
    tr, te = float("nan"), float("nan")
    for line in output.strip().splitlines():
        if "train_acc=" in line and "test_acc=" in line:
            tokens = line.split()
            for tok in tokens:
                if tok.startswith("train_acc="):
                    tr = float(tok.split("=")[1])
                if tok.startswith("test_acc="):
                    te = float(tok.split("=")[1])
    return tr, te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["cnn", "mnist-rnn"], default="cnn")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--budget_epochs", type=int, default=15)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--full_grid", action="store_true")
    args = parser.parse_args()

    if args.out_csv is None:
        args.out_csv = (
            "./outputs/csv/search_results_cnn.csv"
            if args.task == "cnn"
            else "./outputs/csv/search_results_mnist.csv"
        )

    results: List[Dict[str, Any]] = []

    if args.task == "cnn":
        if args.full_grid:
            lr_values = [0.02, 0.01, 0.005, 0.001]
            activations = ["tanh", "relu"]
            momenta = [0.9, 0.95]
            optimizers = ["sgd", "adam"]
            configs = []
            for opt in optimizers:
                for lr in lr_values:
                    for act in activations:
                        for mom in momenta:
                            configs.append(
                                {
                                    "optimizer": opt,
                                    "lr": lr,
                                    "batch_size": 128,
                                    "activation": act,
                                    "momentum": mom,
                                }
                            )
            if len(configs) > 10:
                configs = configs[:10]
        else:
            configs = [
                {"optimizer": "sgd", "lr": 0.02, "batch_size": 128, "activation": "relu", "momentum": 0.9},
                {"optimizer": "sgd", "lr": 0.02, "batch_size": 128, "activation": "relu", "momentum": 0.95},
                {"optimizer": "sgd", "lr": 0.02, "batch_size": 128, "activation": "tanh", "momentum": 0.9},
                {"optimizer": "sgd", "lr": 0.02, "batch_size": 128, "activation": "tanh", "momentum": 0.95},
                {"optimizer": "sgd", "lr": 0.01, "batch_size": 128, "activation": "relu", "momentum": 0.9},
                {"optimizer": "sgd", "lr": 0.01, "batch_size": 128, "activation": "tanh", "momentum": 0.9},
                {"optimizer": "sgd", "lr": 0.005, "batch_size": 128, "activation": "relu", "momentum": 0.9},
                {"optimizer": "sgd", "lr": 0.005, "batch_size": 128, "activation": "tanh", "momentum": 0.9},
                {"optimizer": "adam", "lr": 0.001, "batch_size": 128, "activation": "relu", "momentum": 0.9},
                {"optimizer": "adam", "lr": 0.001, "batch_size": 128, "activation": "tanh", "momentum": 0.9},
            ]
        total = len(configs)
        cumulative = 0.0
        for idx, cfg in enumerate(configs, start=1):
            print(f"[sweep][{idx}/{total}] launching CNN run with {cfg}")
            cmd = [
                sys.executable,
                os.path.join("src", "cnn_cifar10_lenet.py"),
                "--epochs",
                str(args.budget_epochs),
                "--optimizer",
                cfg["optimizer"],
                "--lr",
                str(cfg["lr"]),
                "--batch_size",
                str(cfg["batch_size"]),
                "--activation",
                cfg["activation"],
                "--momentum",
                str(cfg["momentum"]),
                "--data_root",
                args.data_root,
                "--outdir",
                "./outputs",
                "--num_workers",
                str(args.num_workers),
                "--no_tqdm",
            ]
            if args.limit_train:
                cmd += ["--limit_train", str(args.limit_train)]
            if args.limit_test:
                cmd += ["--limit_test", str(args.limit_test)]
            code, dur, output = run_cmd(cmd)
            cumulative += dur
            tr_acc, te_acc = parse_metrics(output)
            results.append(
                {
                    "task": "cnn",
                    **cfg,
                    "epochs": args.budget_epochs,
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "duration_sec": dur,
                    "retcode": code,
                }
            )
            print(
                f"[sweep][{idx}/{total}] finished CNN run (test_acc={te_acc:.4f}, duration={dur:.1f}s, cumulative={cumulative:.1f}s, retcode={code})"
            )
    else:
        configs: List[Dict[str, Any]] = [
            {"rnn_type": "lstm", "hidden_size": 256, "num_layers": 2, "bidirectional": True, "dropout": 0.2, "lr": 0.001},
            {"rnn_type": "gru", "hidden_size": 256, "num_layers": 2, "bidirectional": True, "dropout": 0.2, "lr": 0.001},
            {"rnn_type": "lstm", "hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.2, "lr": 0.001},
            {"rnn_type": "gru", "hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.2, "lr": 0.001},
            {"rnn_type": "lstm", "hidden_size": 256, "num_layers": 1, "bidirectional": False, "dropout": 0.0, "lr": 0.001},
            {"rnn_type": "gru", "hidden_size": 256, "num_layers": 1, "bidirectional": False, "dropout": 0.0, "lr": 0.001},
            {"rnn_type": "lstm", "hidden_size": 128, "num_layers": 1, "bidirectional": False, "dropout": 0.0, "lr": 0.001},
            {"rnn_type": "gru", "hidden_size": 128, "num_layers": 1, "bidirectional": False, "dropout": 0.0, "lr": 0.001},
            {"rnn_type": "rnn", "hidden_size": 256, "num_layers": 2, "bidirectional": False, "dropout": 0.2, "lr": 0.001},
            {"rnn_type": "rnn", "hidden_size": 128, "num_layers": 1, "bidirectional": False, "dropout": 0.0, "lr": 0.001},
        ]
        total = len(configs)
        cumulative = 0.0
        for idx, cfg in enumerate(configs, start=1):
            print(f"[sweep][{idx}/{total}] launching MNIST run with {cfg}")
            cmd = [
                sys.executable,
                os.path.join("src", "rnn_mnist.py"),
                "--epochs",
                str(args.budget_epochs),
                "--rnn_type",
                cfg["rnn_type"],
                "--lr",
                str(cfg["lr"]),
                "--hidden_size",
                str(cfg["hidden_size"]),
                "--num_layers",
                str(cfg["num_layers"]),
                "--batch_size",
                "128",
                "--data_root",
                args.data_root,
                "--outdir",
                "./outputs",
                "--num_workers",
                str(args.num_workers),
                "--optimizer",
                "adam",
                "--dropout",
                str(cfg["dropout"]),
                "--no_tqdm",
            ]
            if cfg["bidirectional"]:
                cmd.append("--bidirectional")
            if args.limit_train:
                cmd += ["--limit_train", str(args.limit_train)]
            if args.limit_test:
                cmd += ["--limit_test", str(args.limit_test)]
            code, dur, output = run_cmd(cmd)
            cumulative += dur
            tr_acc, te_acc = parse_metrics(output)
            results.append(
                {
                    "task": "mnist",
                    **cfg,
                    "batch_size": 128,
                    "optimizer": "adam",
                    "epochs": args.budget_epochs,
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "duration_sec": dur,
                    "retcode": code,
                }
            )
            print(
                f"[sweep][{idx}/{total}] finished MNIST run (test_acc={te_acc:.4f}, duration={dur:.1f}s, cumulative={cumulative:.1f}s, retcode={code})"
            )

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved search results to", args.out_csv)


if __name__ == "__main__":
    main()
