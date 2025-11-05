# Dev Sanghvi (ds221)

"""Utilities (seed + dataloaders) imported by cnn_cifar10_lenet.py and rnn_mnist.py."""

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int = 576) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_with_onehot(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    oh = torch.nn.functional.one_hot(torch.tensor(targets), num_classes=10).float()
    return imgs, oh


def collate_seq(batch):
    imgs, targets = zip(*batch)
    x = torch.stack(imgs, dim=0).squeeze(1)
    y = torch.tensor(targets, dtype=torch.long)
    return x, y


def get_cifar10_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    one_hot: bool = True,
    limit_train: int = None,
    limit_test: int = None,
    log: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    if log:
        print("[data] CIFAR-10: RGB 32x32 + normalize", flush=True)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    pin_memory = torch.cuda.is_available()
    try:
        train_set = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=True)
        test_set = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to load CIFAR-10. Place dataset under data_root if offline. " f"Original error: {repr(e)}"
        )
    if limit_train is not None:
        train_set = Subset(train_set, range(min(limit_train, len(train_set))))
    if limit_test is not None:
        test_set = Subset(test_set, range(min(limit_test, len(test_set))))
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_with_onehot if one_hot else None,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def get_mnist_sequence_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    limit_train: int = None,
    limit_test: int = None,
    log: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    if log:
        print("[data] MNIST: ToTensor", flush=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    pin_memory = torch.cuda.is_available()
    try:
        train_set = datasets.MNIST(root=data_root, train=True, transform=tfm, download=True)
        test_set = datasets.MNIST(root=data_root, train=False, transform=tfm, download=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to load MNIST. Place dataset under data_root if offline. " f"Original error: {repr(e)}"
        )
    if limit_train is not None:
        train_set = Subset(train_set, range(min(limit_train, len(train_set))))
    if limit_test is not None:
        test_set = Subset(test_set, range(min(limit_test, len(test_set))))
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_seq,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
