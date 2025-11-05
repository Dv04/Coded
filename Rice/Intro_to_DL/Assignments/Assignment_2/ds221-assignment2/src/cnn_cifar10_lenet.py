# Dev Sanghvi (ds221)

"""CNN training script for CIFAR-10 (LeNet).
Runs standalone: python src/cnn_cifar10_lenet.py --epochs 15 --batch_size 128 --lr 0.02 --optimizer sgd"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from typing import Optional
from tqdm import tqdm

from utils import set_seed, device_auto, get_cifar10_loaders
from models.lenet import LeNet5
from viz import plot_curves, visualize_conv1_filters, plot_activation_stats, plot_confusion

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _generate_boxplots_and_occlusion(model, test_loader, device, outdir):
    from viz import collect_conv_activations, plot_activation_boxplots, save_occlusion_grid

    per_layer = collect_conv_activations(
        model,
        test_loader,
        layers=[model.conv1, model.conv2, getattr(model, "conv3", None)],
        max_batches=10,
        device=device,
    )
    plot_activation_boxplots(per_layer, os.path.join(outdir, "figures"))
    dataset = getattr(test_loader, "dataset", None)
    if dataset is not None:
        n = min(6, len(dataset))
        idx_list = list(range(n))
        save_occlusion_grid(
            model,
            dataset,
            idx_list,
            os.path.join(outdir, "figures", "cnn_occlusion_grid.png"),
            device=device,
        )


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
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y_idx)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc_sum += (pred == y_idx).float().sum().item()
            loss_sum += loss.item() * x.size(0)
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
    y_true, y_pred = [], []
    iterator = (
        tqdm(loader, desc=desc, leave=False, mininterval=0.2, dynamic_ncols=True)
        if use_tqdm
        else loader
    )
    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        y_true.extend(y_idx.cpu().tolist())
        y_pred.extend(preds.tolist())
        acc_sum += (preds == y_idx.cpu()).float().sum().item()
        n += x.size(0)
    return acc_sum / max(1, n), np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
    parser.add_argument("--seed", type=int, default=576)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="./outputs")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable progress bars.")
    parser.add_argument("--history_out", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)

    device = device_auto()
    set_seed(args.seed)

    train_loader, test_loader = get_cifar10_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        one_hot=True,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )

    model = LeNet5(activation=args.activation).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=5e-4,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_state, best_acc = None, -1.0
    use_tqdm = not args.no_tqdm and sys.stderr.isatty()

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_tqdm=use_tqdm,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        test_acc, _, _ = evaluate(
            model,
            test_loader,
            device,
            use_tqdm=use_tqdm,
            desc=f"Eval {epoch}/{args.epochs}" if use_tqdm else "Eval",
        )
        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.outdir, "checkpoints", "lenet5_best.pt"))
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f} lr={scheduler.get_last_lr()[0]:.5f}"
        )
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    if args.history_out:
        hist_dir = os.path.dirname(args.history_out)
        if hist_dir:
            os.makedirs(hist_dir, exist_ok=True)
        torch.save(history, args.history_out)
    viz_model = LeNet5(activation=args.activation).to(device)
    viz_model.load_state_dict(model.state_dict())
    viz_model.eval()

    with torch.no_grad():
        prefix = os.path.join(args.outdir, "figures", "cnn_lenet")
        plot_curves(history, prefix)
        visualize_conv1_filters(
            viz_model.conv1.weight,
            os.path.join(args.outdir, "figures", "cnn_lenet_filters.png"),
        )
        x_sample, _ = next(iter(test_loader))
        x_sample = x_sample.to(device)
        logits, acts = viz_model(x_sample, return_activations=True)
        plot_activation_stats(acts, os.path.join(args.outdir, "figures", "cnn_lenet"))
        _, y_true, y_pred = evaluate(
            viz_model,
            test_loader,
            device,
            use_tqdm=False,
        )
        plot_confusion(
            y_true,
            y_pred,
            CLASSES,
            os.path.join(args.outdir, "figures", "cnn_lenet_confusion.png"),
        )
        _generate_boxplots_and_occlusion(viz_model, test_loader, device, args.outdir)


if __name__ == "__main__":
    main()
