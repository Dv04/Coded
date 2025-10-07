#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELEC 576 / COMP 576 — Assignment 1 (Part 2)
Simple 5-layer DCN on MNIST with TensorBoard monitoring.
This script is self-contained; it implements the "starter skeleton" fully.
Author: <YOUR NAME / NETID HERE>
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


# ------------------------- Model -------------------------
class Net(nn.Module):
    def __init__(
        self, activation: str = "relu", dropout_p: float = 0.5, init: str = "kaiming"
    ):
        super().__init__()
        # conv1(5-5-1-32) - ReLU - maxpool(2-2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.relu1 = self._make_activation(activation)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2(5-5-32-64) - ReLU - maxpool(2-2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.relu2 = self._make_activation(activation)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fc(1024) - ReLU - DropOut(0.5) - Softmax(10)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.relu3 = self._make_activation(activation)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(
            1024, 10
        )  # logits -> CrossEntropyLoss applies softmax internally

        self._init_weights(init)

        # Placeholders for monitoring via hooks
        self._activations = {}
        self._preacts = {}
        self._register_hooks()

    def _make_activation(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=False)
        elif name == "leaky_relu":
            return nn.LeakyReLU(inplace=False)
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def _init_weights(self, method: str = "kaiming"):
        def init_module(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if method.lower() in ["xavier", "glorot"]:
                    nn.init.xavier_uniform_(m.weight)
                elif method.lower() in ["kaiming", "he"]:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    # default PyTorch init
                    pass
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_module)

    def _register_hooks(self):
        # We log pre-activations ("z") and post-ReLU activations ("a"), plus post-pooling outputs.
        def make_hook(name, container):
            def hook(module, inp, out):
                # Avoid storing huge tensors: take a detached copy of a random subset
                t = out.detach()
                if t.numel() > 8192:
                    t = t.flatten()[:8192].reshape(-1)  # sample first 8k values
                container[name] = t

            return hook

        # Pre-activations are outputs of conv/linear before non-linearity
        self.conv1.register_forward_hook(make_hook("z/conv1", self._preacts))
        self.conv2.register_forward_hook(make_hook("z/conv2", self._preacts))
        self.fc1.register_forward_hook(make_hook("z/fc1", self._preacts))
        self.fc2.register_forward_hook(make_hook("z/fc2", self._preacts))

        # Activations after non-linearities
        self.relu1.register_forward_hook(make_hook("a/relu1", self._activations))
        self.relu2.register_forward_hook(make_hook("a/relu2", self._activations))
        self.relu3.register_forward_hook(make_hook("a/relu3", self._activations))

        # Activations after max-pooling
        self.pool1.register_forward_hook(make_hook("a/pool1", self._activations))
        self.pool2.register_forward_hook(make_hook("a/pool2", self._activations))

    def forward(self, x):
        # conv1 -> act -> pool -> conv2 -> act -> pool -> flatten -> fc1 -> act -> drop -> fc2
        x = self.conv1(x)
        z1 = x
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        z2 = x
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        z3 = x
        x = self.relu3(x)
        x = self.drop(x)
        x = self.fc2(x)
        z4 = x
        return x  # logits


# ------------------------- Data -------------------------
def get_dataloaders(batch_size=128, val_size=5000, workers=2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trainval = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_size = len(trainval) - val_size
    train, val = random_split(
        trainval, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader


# ------------------------- Training -------------------------
def build_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr)
    elif name == "momentum":
        return optim.SGD(params, lr=lr, momentum=momentum)
    elif name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError("Unknown optimizer")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        args.batch_size, args.val_size, args.workers
    )

    model = Net(activation=args.activation, dropout_p=args.dropout, init=args.init).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # TensorBoard
    run_name = f"MNIST_{args.activation}_{args.optimizer}_{args.init}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    global_step = 0
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            # Forward
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Monitor training loss
            if global_step % args.log_every == 0:
                writer.add_scalar("loss/train", loss.item(), global_step)

                # (b) Log statistics for weights and biases
                for name, p in model.named_parameters():
                    # Histograms
                    writer.add_histogram(
                        f"params/{name}", p.detach().cpu(), global_step
                    )
                    # Summary stats
                    writer.add_scalar(f"stats/{name}_min", p.min().item(), global_step)
                    writer.add_scalar(f"stats/{name}_max", p.max().item(), global_step)
                    writer.add_scalar(
                        f"stats/{name}_mean", p.mean().item(), global_step
                    )
                    writer.add_scalar(f"stats/{name}_std", p.std().item(), global_step)

                # (b) Log pre-activations and activations
                for k, t in model._preacts.items():
                    writer.add_histogram(k, t.cpu(), global_step)
                    writer.add_scalar(f"stats/{k}_min", t.min().item(), global_step)
                    writer.add_scalar(f"stats/{k}_max", t.max().item(), global_step)
                    writer.add_scalar(f"stats/{k}_mean", t.mean().item(), global_step)
                    writer.add_scalar(f"stats/{k}_std", t.std().item(), global_step)
                for k, t in model._activations.items():
                    writer.add_histogram(k, t.cpu(), global_step)
                    writer.add_scalar(f"stats/{k}_min", t.min().item(), global_step)
                    writer.add_scalar(f"stats/{k}_max", t.max().item(), global_step)
                    writer.add_scalar(f"stats/{k}_mean", t.mean().item(), global_step)
                    writer.add_scalar(f"stats/{k}_std", t.std().item(), global_step)

            global_step += 1

        # End epoch: evaluate
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("acc/test", test_acc, epoch)

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join("checkpoints", f"best_{run_name}.pt")
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | Train loss {loss.item():.4f} | Val acc {val_acc:.4f} | Test acc {test_acc:.4f}"
        )

    writer.close()
    print("Training complete. Best val acc:", best_val)


def parse_args():
    p = argparse.ArgumentParser(
        description="MNIST DCN Training with TensorBoard Monitoring"
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "adagrad", "adam", "rmsprop"],
    )
    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "tanh", "sigmoid"],
    )
    p.add_argument(
        "--init",
        type=str,
        default="kaiming",
        choices=["kaiming", "xavier", "glorot", "default"],
    )
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="iterations between TensorBoard logging (≈ part b)",
    )
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
