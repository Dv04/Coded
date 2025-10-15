#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from pathlib import Path

# Resolve project root relative to this file
PROJ_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = PROJ_ROOT / "report" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=4824):
    np.random.seed(seed)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def one_hot(y, C):
    out = np.zeros((y.shape[0], C), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def act(z, name):
    name = name.lower()
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("Unknown activation")


def dact(z, name):
    name = name.lower()
    if name == "relu":
        return (z > 0.0).astype(z.dtype)
    if name == "tanh":
        a = np.tanh(z)
        return 1.0 - a**2
    if name == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-z))
        return s * (1.0 - s)
    raise ValueError("Unknown activation")


class DeepMLP:
    def __init__(
        self,
        input_dim=2,
        output_dim=2,
        n_hidden_layers=3,
        hidden_size=16,
        activation="relu",
        reg_lambda=1e-3,
        lr=0.05,
        seed=4824,
    ):
        set_seed(seed)
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.lr = lr
        sizes = [input_dim] + [hidden_size] * n_hidden_layers + [output_dim]
        self.L = len(sizes) - 1
        self.W = [0.01 * np.random.randn(sizes[l], sizes[l + 1]) for l in range(self.L)]
        self.b = [np.zeros((1, sizes[l + 1])) for l in range(self.L)]
        self.z = [None] * self.L
        self.a = [None] * (self.L + 1)

    def forward(self, X):
        self.a[0] = X
        for l in range(self.L - 1):
            self.z[l] = self.a[l].dot(self.W[l]) + self.b[l]
            self.a[l + 1] = act(self.z[l], self.activation)
        self.z[self.L - 1] = (
            self.a[self.L - 1].dot(self.W[self.L - 1]) + self.b[self.L - 1]
        )
        self.probs = softmax(self.z[self.L - 1])
        return self.probs

    def loss(self, X, y):
        N = X.shape[0]
        probs = self.forward(X)
        yoh = one_hot(y, self.b[-1].shape[1])
        ce = -np.sum(yoh * np.log(probs + 1e-12)) / N
        reg = 0.5 * self.reg_lambda * sum([np.sum(W**2) for W in self.W])
        return ce + reg

    def backward(self, X, y):
        N = X.shape[0]
        yoh = one_hot(y, self.b[-1].shape[1])
        delta = (self.probs - yoh) / N
        dW = [None] * self.L
        db = [None] * self.L
        dW[self.L - 1] = (
            self.a[self.L - 1].T.dot(delta) + self.reg_lambda * self.W[self.L - 1]
        )
        db[self.L - 1] = np.sum(delta, axis=0, keepdims=True)
        for l in reversed(range(self.L - 1)):
            delta = delta.dot(self.W[l + 1].T) * dact(self.z[l], self.activation)
            dW[l] = self.a[l].T.dot(delta) + self.reg_lambda * self.W[l]
            db[l] = np.sum(delta, axis=0, keepdims=True)
        return dW, db

    def step(self, grads):
        dW, db = grads
        for l in range(self.L):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]

    def fit(self, X, y, iters=8000, print_every=2000, verbose_prefix=""):
        for i in range(1, iters + 1):
            _ = self.forward(X)
            self.step(self.backward(X, y))
            if i % print_every == 0:
                print(f"{verbose_prefix}Iter {i}: loss={self.loss(X,y):.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def compute_accuracy(model: DeepMLP, X: np.ndarray, y: np.ndarray) -> float:
    """Return mean accuracy on (X, y)."""
    return float(np.mean(model.predict(X) == y))


def plot_decision_boundary(predict_fn, X, y, title, savepath):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=25, edgecolors="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(savepath, dpi=180)
    plt.close()


def save_dataset_scatter(X, y, title, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=25, edgecolors="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_toy(name="moons", n=400, noise=0.2, seed=4824):
    set_seed(seed)
    if name == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n, noise=noise, random_state=seed, factor=0.5)
    elif name == "blobs":
        X, y = make_blobs(n_samples=n, centers=3, cluster_std=1.0, random_state=seed)
        y = (y - y.min()).astype(int)
    else:
        raise ValueError("Unknown dataset")
    return X.astype(np.float64), y.astype(np.int64)


def experiment():
    print("\n=== Dataset: moons (n=400, noise=0.2) ===")
    X, y = load_toy("moons", n=400, noise=0.2, seed=4824)
    # Depth × width grid (activation exposed)
    for L in [1, 2, 4, 6]:
        for H in [4, 16, 64]:
            model = DeepMLP(
                input_dim=2,
                output_dim=2,
                n_hidden_layers=L,
                hidden_size=H,
                activation="relu",
                reg_lambda=1e-3,
                lr=0.05,
                seed=0,
            )
            ctx = f"[moons] L={L}, H={H}, act=ReLU | "
            print(
                f"\n=== Training {ctx}iters=6000, lr={model.lr}, reg={model.reg_lambda} ==="
            )
            model.fit(X, y, iters=6000, print_every=1000, verbose_prefix=ctx)
            final_loss = model.loss(X, y)
            acc = compute_accuracy(model, X, y)
            print(f"--- Final: loss={final_loss:.4f}, train acc={acc:.3f}")
            title = f"DeepNN — L={L}, H={H}, act=ReLU"
            save = FIGS_DIR / f"deep_decision_L{L}_H{H}.png"
            plot_decision_boundary(lambda xx: model.predict(xx), X, y, title, save)
            print(f"✓ Saved: {save}")

    for ds in ["circles", "blobs"]:
        Xd, yd = load_toy(ds, n=400, noise=0.2, seed=4824)
        model = DeepMLP(
            input_dim=2,
            output_dim=(yd.max() + 1),
            n_hidden_layers=3,
            hidden_size=32,
            activation="tanh",
            reg_lambda=1e-3,
            lr=0.05,
            seed=7,
        )
        print(f"\n=== Dataset: {ds} (n=400, noise=0.2) ===")
        ctx = f"[{ds}] L=3, H=32, act=Tanh | "
        print(
            f"=== Training {ctx}iters=6000, lr={model.lr}, reg={model.reg_lambda} ==="
        )
        model.fit(Xd, yd, iters=6000, print_every=1000, verbose_prefix=ctx)
        final_loss = model.loss(Xd, yd)
        acc = compute_accuracy(model, Xd, yd)
        print(f"--- Final: loss={final_loss:.4f}, train acc={acc:.3f}")
        title = f"DeepNN on {ds} — L=3, H=32, act=Tanh"
        save = FIGS_DIR / f"deep_{ds}_L3_H32.png"
        plot_decision_boundary(lambda xx: model.predict(xx), Xd, yd, title, save)
        print(f"✓ Saved: {save}")


if __name__ == "__main__":
    experiment()
