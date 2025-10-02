#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELEC 576 / COMP 576 — Assignment 1 (Part 1)
Simple 3-layer neural network with manual backprop on Make-Moons.
Author: <YOUR NAME / NETID HERE>
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Dict

# Optional: scikit-learn only for data generation/standardization
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=4, suppress=True)


def generate_data(n_samples: int = 200, noise: float = 0.2, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Make-Moons dataset (2D, 2 classes).
    Returns:
        X: shape (N,2), y: shape (N,)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float64), y.astype(np.int64)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode a 1D integer label vector."""
    N = y.shape[0]
    out = np.zeros((N, num_classes), dtype=np.float64)
    out[np.arange(N), y] = 1.0
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax."""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_scores = np.exp(z_shift)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def plot_decision_boundary(predict_fn: Callable[[np.ndarray], np.ndarray],
                           X: np.ndarray,
                           y: np.ndarray,
                           title: str = "",
                           savepath: Optional[str] = None) -> None:
    """Plot 2D decision boundary for a classifier with predict(X)->labels."""
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=180)
        plt.close()
    else:
        plt.show()


@dataclass
class NNConfig:
    input_dim: int = 2
    hidden_dim: int = 3
    output_dim: int = 2
    activation: str = "Tanh"      # {'Tanh','Sigmoid','ReLU'}
    learning_rate: float = 0.01
    reg_lambda: float = 0.0       # L2 regularization (not required in Part 1; keep at 0.0)
    seed: int = 42


class NeuralNetwork:
    """Three-layer (input - hidden - output) neural network with manual backprop."""

    def __init__(self, config: NNConfig):
        self.config = config
        np.random.seed(config.seed)

        # Parameter initialization: small random weights, zero bias
        self.W1 = 0.01 * np.random.randn(config.input_dim, config.hidden_dim)
        self.b1 = np.zeros((1, config.hidden_dim))
        self.W2 = 0.01 * np.random.randn(config.hidden_dim, config.output_dim)
        self.b2 = np.zeros((1, config.output_dim))

        # caches
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.probs = None

    # ------------------------- Activation + Derivatives -------------------------
    @staticmethod
    def actFun(z: np.ndarray, type: str) -> np.ndarray:
        if type == 'Tanh':
            return np.tanh(z)
        elif type == 'Sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif type == 'ReLU':
            return np.maximum(0.0, z)
        else:
            raise ValueError(f"Unsupported activation: {type}")

    @staticmethod
    def diff_actFun(z: np.ndarray, type: str) -> np.ndarray:
        if type == 'Tanh':
            a = np.tanh(z)
            return 1.0 - a**2
        elif type == 'Sigmoid':
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        elif type == 'ReLU':
            return (z > 0).astype(z.dtype)
        else:
            raise ValueError(f"Unsupported activation: {type}")

    # ------------------------------ Forward pass --------------------------------
    def feedforward(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for inputs X."""
        self.z1 = X.dot(self.W1) + self.b1          # (N,H)
        self.a1 = self.actFun(self.z1, self.config.activation)  # (N,H)
        self.z2 = self.a1.dot(self.W2) + self.b2    # (N,C)
        self.probs = softmax(self.z2)               # (N,C)
        return self.probs

    # ------------------------------ Loss function -------------------------------
    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy loss (averaged over N) + optional L2 regularization."""
        N = X.shape[0]
        probs = self.feedforward(X)
        y_onehot = one_hot(y, self.config.output_dim)
        # Avoid log(0)
        core_loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
        if self.config.reg_lambda > 0.0:
            reg = 0.5 * self.config.reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
        else:
            reg = 0.0
        return core_loss + reg

    # -------------------------------- Backprop ----------------------------------
    def backprop(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients for W1, b1, W2, b2 (averaged over N) using backprop."""
        N = X.shape[0]
        # Forward already computed in calculate_loss / feedforward
        probs = self.probs if self.probs is not None else self.feedforward(X)
        y_onehot = one_hot(y, self.config.output_dim)

        # dL/dz2 for softmax+cross-entropy: (probs - y_onehot)/N
        delta3 = (probs - y_onehot) / N  # (N,C)

        dW2 = self.a1.T.dot(delta3)      # (H,N) x (N,C) -> (H,C)
        db2 = np.sum(delta3, axis=0, keepdims=True)  # (1,C)

        # Backprop through hidden
        delta2 = delta3.dot(self.W2.T) * self.diff_actFun(self.z1, self.config.activation)  # (N,H)
        dW1 = X.T.dot(delta2)            # (D,N) x (N,H) -> (D,H)
        db1 = np.sum(delta2, axis=0, keepdims=True)  # (1,H)

        # L2
        if self.config.reg_lambda > 0.0:
            dW2 += self.config.reg_lambda * self.W2
            dW1 += self.config.reg_lambda * self.W1

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """SGD parameter update."""
        self.W1 -= self.config.learning_rate * grads['dW1']
        self.b1 -= self.config.learning_rate * grads['db1']
        self.W2 -= self.config.learning_rate * grads['dW2']
        self.b2 -= self.config.learning_rate * grads['db2']

    def fit_model(self, X: np.ndarray, y: np.ndarray, num_passes: int = 10000,
                  print_every: int = 1000) -> None:
        for i in range(1, num_passes + 1):
            # Forward + loss
            _ = self.feedforward(X)
            # Backward
            grads = self.backprop(X, y)
            # Update
            self.step(grads)

            if (i % print_every) == 0:
                loss = self.calculate_loss(X, y)
                print(f"Iteration {i:5d} | loss = {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.feedforward(X)
        return np.argmax(probs, axis=1)


def main():
    # -------- (a) Generate and visualize Make-Moons dataset --------
    X, y = generate_data(n_samples=200, noise=0.2, random_state=0)
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("Make-Moons dataset")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig("report/figs/moons_scatter.png", dpi=180)
    plt.close()

    # Split for sanity (train only is fine for this toy example)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    # -------- (e.1) Train with different activations --------
    activations = ["Tanh", "Sigmoid", "ReLU"]
    for act in activations:
        cfg = NNConfig(hidden_dim=3, activation=act, learning_rate=0.05, reg_lambda=0.0, seed=42)
        nn = NeuralNetwork(cfg)
        nn.fit_model(Xtr, ytr, num_passes=5000, print_every=1000)

        title = f"Decision boundary — act={act}, H=3"
        savepath = f"report/figs/decision_boundary_{act.lower()}_H3.png"
        plot_decision_boundary(lambda xx: nn.predict(xx), X, y, title, savepath)

    # -------- (e.2) Increase hidden units and retrain (Tanh) --------
    for H in [3, 10, 50]:
        cfg = NNConfig(hidden_dim=H, activation="Tanh", learning_rate=0.05, reg_lambda=0.0, seed=42)
        nn = NeuralNetwork(cfg)
        nn.fit_model(Xtr, ytr, num_passes=5000, print_every=1000)

        title = f"Decision boundary — act=Tanh, H={H}"
        savepath = f"report/figs/decision_boundary_tanh_H{H}.png"
        plot_decision_boundary(lambda xx: nn.predict(xx), X, y, title, savepath)


if __name__ == "__main__":
    main()
