# Dev Sanghvi (ds221)

"""MNIST sequence (RNN/GRU/LSTM) trainer.
Runs standalone: python src/rnn_mnist.py --rnn_type gru --epochs 10 --hidden_size 256 --lr 1e-3"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from typing import Optional
from tqdm import tqdm

from utils import set_seed, device_auto, get_mnist_sequence_loaders
from models.rnn_models import RNNSequenceClassifier
from viz import plot_curves


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    *,
    use_tqdm=False,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    iterator = loader
    if use_tqdm:
        desc = f"Epoch {epoch}/{total_epochs}" if epoch and total_epochs else "Train"
        iterator = tqdm(loader, desc=desc, leave=False, mininterval=0.2, dynamic_ncols=True)
    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        acc_sum += (logits.argmax(dim=1) == y).float().sum().item()
        n += x.size(0)
    return loss_sum / max(1, n), acc_sum / max(1, n)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    use_tqdm=False,
    desc="Eval",
):
    model.eval()
    acc_sum, n = 0.0, 0
    iterator = (
        tqdm(loader, desc=desc, leave=False, mininterval=0.2, dynamic_ncols=True)
        if use_tqdm
        else loader
    )
    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        acc_sum += (preds == y).float().sum().item()
        n += x.size(0)
    return acc_sum / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--rnn_type", type=str, default="rnn", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--seed", type=int, default=576)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="./outputs")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)

    device = device_auto()
    set_seed(args.seed)

    train_loader, test_loader = get_mnist_sequence_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )

    model = RNNSequenceClassifier(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=args.lr)
        if args.optimizer == "adam"
        else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    )

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_acc, best_state = -1.0, None
    use_tqdm = not args.no_tqdm and sys.stderr.isatty()

    for epoch in range(1, args.epochs + 1):
        loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_tqdm=use_tqdm,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        te_acc = evaluate(
            model,
            test_loader,
            device,
            use_tqdm=use_tqdm,
            desc=f"Eval {epoch}/{args.epochs}" if use_tqdm else "Eval",
        )
        history["train_loss"].append(loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(
            f"[{args.rnn_type.upper()}] Epoch {epoch:03d} | loss={loss:.4f} train_acc={tr_acc:.4f} test_acc={te_acc:.4f}"
        )
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt_path = os.path.join(args.outdir, "checkpoints", f"mnist_{args.rnn_type}_best.pt")
    torch.save(best_state, ckpt_path)

    prefix = os.path.join(args.outdir, "figures", f"mnist_{args.rnn_type}")
    plot_curves(history, prefix)


if __name__ == "__main__":
    main()
