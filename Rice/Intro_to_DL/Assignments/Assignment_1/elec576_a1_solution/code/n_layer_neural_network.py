#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELEC 576 / COMP 576 — Assignment 1 (Part 1f)
Deeper fully-connected network of n layers with L2 regularization.
Author: <YOUR NAME / NETID HERE>
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=4, suppress=True)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(y: np.ndarray, C: int) -> np.ndarray:
    N = y.shape[0]
    out = np.zeros((N, C), dtype=np.float64)
    out[np.arange(N), y] = 1.0
    return out


def act(z: np.ndarray, name: str) -> np.ndarray:
    if name == "ReLU":
        return np.maximum(0., z)
    elif name == "Tanh":
        return np.tanh(z)
    elif name == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"Unsupported activation: {name}")


def dact(z: np.ndarray, name: str) -> np.ndarray:
    if name == "ReLU":
        return (z > 0.).astype(z.dtype)
    elif name == "Tanh":
        a = np.tanh(z)
        return 1. - a**2
    elif name == "Sigmoid":
        s = 1.0 / (1.0 + np.exp(-z))
        return s * (1.0 - s)
    else:
        raise ValueError(f"Unsupported activation: {name}")


@dataclass
class DeepConfig:
    input_dim: int = 2
    output_dim: int = 2
    n_hidden_layers: int = 3     # number of hidden layers (>=1)
    hidden_size: int = 16        # width of each hidden layer
    activation: str = "ReLU"
    reg_lambda: float = 1e-3
    learning_rate: float = 0.05
    seed: int = 1


class DeepNeuralNetwork:
    """n-layer MLP with identical hidden sizes and chosen activation; softmax output."""
    def __init__(self, cfg: DeepConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)

        sizes = [cfg.input_dim] + [cfg.hidden_size] * cfg.n_hidden_layers + [cfg.output_dim]
        self.L = len(sizes) - 1  # number of weight layers

        self.W = [0.01 * np.random.randn(sizes[l], sizes[l+1]) for l in range(self.L)]
        self.b = [np.zeros((1, sizes[l+1])) for l in range(self.L)]

        # caches
        self.z = [None] * self.L
        self.a = [None] * (self.L + 1)  # include input as a[0]
        self.probs = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.a[0] = X
        for l in range(self.L - 1):  # hidden layers
            self.z[l] = self.a[l].dot(self.W[l]) + self.b[l]
            self.a[l+1] = act(self.z[l], self.cfg.activation)
        # output (linear -> softmax)
        self.z[self.L - 1] = self.a[self.L - 1].dot(self.W[self.L - 1]) + self.b[self.L - 1]
        self.probs = softmax(self.z[self.L - 1])
        return self.probs

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        N = X.shape[0]
        probs = self.forward(X)
        y_onehot = one_hot(y, self.cfg.output_dim)
        ce = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
        reg = 0.5 * self.cfg.reg_lambda * sum([np.sum(W**2) for W in self.W])
        return ce + reg

    def backward(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        probs = self.probs if self.probs is not None else self.forward(X)
        y_onehot = one_hot(y, self.cfg.output_dim)

        dW = [None] * self.L
        db = [None] * self.L

        # output layer gradient
        delta = (probs - y_onehot) / N  # (N,C)
        dW[self.L - 1] = self.a[self.L - 1].T.dot(delta) + self.cfg.reg_lambda * self.W[self.L - 1]
        db[self.L - 1] = np.sum(delta, axis=0, keepdims=True)

        # backprop through hidden layers
        for l in reversed(range(self.L - 1)):
            delta = delta.dot(self.W[l + 1].T) * dact(self.z[l], self.cfg.activation)
            dW[l] = self.a[l].T.dot(delta) + self.cfg.reg_lambda * self.W[l]
            db[l] = np.sum(delta, axis=0, keepdims=True)

        return dW, db

    def step(self, grads):
        dW, db = grads
        for l in range(self.L):
            self.W[l] -= self.cfg.learning_rate * dW[l]
            self.b[l] -= self.cfg.learning_rate * db[l]

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 10000, print_every: int = 1000):
        for i in range(1, iters + 1):
            _ = self.forward(X)
            grads = self.backward(X, y)
            self.step(grads)
            if i % print_every == 0:
                print(f"Iter {i:5d} | loss={self.loss(X, y):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    # ---------- Visualization helpers ----------
    @staticmethod
    def _plot_decision_boundary(predict_fn: Callable[[np.ndarray], np.ndarray],
                                X: np.ndarray, y: np.ndarray,
                                title: str, savepath: Optional[str] = None):
        h = 0.01
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_fn(grid).reshape(xx.shape)

        plt.figure(figsize=(5, 4))
        plt.contourf(xx, yy, Z, alpha=0.6)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
        plt.title(title)
        plt.xlabel("x1"); plt.ylabel("x2")
        if savepath:
            plt.tight_layout(); plt.savefig(savepath, dpi=180); plt.close()
        else:
            plt.show()


def load_toy_dataset(name: str = "moons",
                     n_samples: int = 500,
                     noise: float = 0.2,
                     random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=random_state)
        # convert to 0..C-1 integers
        y = (y - y.min()).astype(int)
    else:
        raise ValueError("Unknown dataset")
    return X.astype(np.float64), y.astype(np.int64)


def experiment():
    # Compare configurations on Make-Moons
    X, y = load_toy_dataset("moons", n_samples=400, noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Vary depth and width
    for L in [1, 2, 4, 6]:  # number of hidden layers
        for H in [4, 16, 64]:
            cfg = DeepConfig(n_hidden_layers=L, hidden_size=H, activation="ReLU",
                             learning_rate=0.05, reg_lambda=1e-3, seed=1)
            model = DeepNeuralNetwork(cfg)
            model.fit(Xtr, ytr, iters=8000, print_every=2000)
            acc_tr = (model.predict(Xtr) == ytr).mean()
            acc_te = (model.predict(Xte) == yte).mean()
            title = f"DeepNN — L={L}, H={H}, act=ReLU (train={acc_tr:.2f}, test={acc_te:.2f})"
            savepath = f"report/figs/deep_decision_L{L}_H{H}.png"
            DeepNeuralNetwork._plot_decision_boundary(lambda xx: model.predict(xx), X, y, title, savepath)

    # Try another dataset
    for ds in ["circles", "blobs"]:
        X, y = load_toy_dataset(ds, n_samples=400, noise=0.2, random_state=1)
        cfg = DeepConfig(n_hidden_layers=3, hidden_size=32, activation="Tanh",
                         learning_rate=0.05, reg_lambda=1e-3, seed=7)
        model = DeepNeuralNetwork(cfg)
        model.fit(X, y, iters=8000, print_every=2000)
        title = f"DeepNN on {ds} — L=3, H=32, act=Tanh"
        savepath = f"report/figs/deep_{ds}_L3_H32.png"
        DeepNeuralNetwork._plot_decision_boundary(lambda xx: model.predict(xx), X, y, title, savepath)


if __name__ == "__main__":
    experiment()
