__author__ = "Dev Sanghvi"
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from typing import Callable, Optional
from pathlib import Path

# Resolve project root relative to this file
PROJ_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = PROJ_ROOT / "report" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- Reproducibility -----------------
def set_seed(seed: int = 4824):
    np.random.seed(seed)


# ----------------- Data + Plot -----------------
def generate_data(n_samples: int = 200, noise: float = 0.20, seed: int = 4824):
    set_seed(seed)
    X, y = datasets.make_moons(n_samples, noise=noise)
    return X, y


def plot_decision_boundary(
    pred_func: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    title: str = "",
    savepath: Optional[str] = None,
):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=25, edgecolors="k")
    if title:
        plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180)
        plt.close()
    else:
        plt.show()


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


# ----------------- Model -----------------
class NeuralNetwork(object):
    """
    3-layer NN (input -> hidden -> output) for Make-Moons with manual backprop.
    """

    def __init__(
        self,
        nn_input_dim: int,
        nn_hidden_dim: int,
        nn_output_dim: int,
        actFun_type: str = "tanh",
        reg_lambda: float = 0.01,
        seed: int = 4824,
    ):
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        set_seed(seed)
        # Stable small init (Xavier-ish)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(
            self.nn_input_dim
        )
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(
            self.nn_hidden_dim
        )
        self.b2 = np.zeros((1, self.nn_output_dim))

        # caches
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.probs = None

    # ---------- activations ----------
    def actFun(self, z, type):
        type = type.lower()
        if type == "tanh":
            return np.tanh(z)
        elif type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        elif type == "relu":
            return np.maximum(0.0, z)
        else:
            raise Exception("Invalid activation function")

    def diff_actFun(self, z, type):
        type = type.lower()
        if type == "tanh":
            a = np.tanh(z)
            return 1.0 - a**2
        elif type == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        elif type == "relu":
            return (z > 0).astype(z.dtype)
        else:
            raise Exception("Invalid activation function")

    # ---------- forward / loss ----------
    def feedforward(self, X, actFun):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        z2s = self.z2 - np.max(self.z2, axis=1, keepdims=True)  # stable softmax
        exp_scores = np.exp(z2s)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # average cross-entropy
        log_likelihood = -np.log(self.probs[range(num_examples), y] + 1e-12)
        data_loss = np.mean(log_likelihood)
        # L2
        data_loss += self.reg_lambda / 2.0 * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss

    # ---------- backward ----------
    def backprop(self, X, y):
        """
        Returns gradients in order: dW1, dW2, db1, db2
        """
        num_examples = len(X)
        y_onehot = np.zeros_like(self.probs)
        y_onehot[np.arange(num_examples), y] = 1.0
        # average grads to match averaged loss
        delta3 = (self.probs - y_onehot) / num_examples

        dW2 = self.a1.T.dot(delta3) + self.reg_lambda * self.W2
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = (delta3.dot(self.W2.T)) * self.diff_actFun(
            self.z1, type=self.actFun_type
        )
        dW1 = X.T.dot(delta2) + self.reg_lambda * self.W1
        db1 = np.sum(delta2, axis=0, keepdims=True)

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=10000, print_loss=True):
        for i in range(num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            dW1, dW2, db1, db2 = self.backprop(X, y)

            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y):.4f}")

    def predict(self, X):
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def visualize_decision_boundary(self, X, y, title="", savepath=None):
        plot_decision_boundary(
            lambda x: self.predict(x), X, y, title=title, savepath=savepath
        )


# ----------------- Auto-run for required plots -----------------
def main():
    # Train + save plots for activation sweep and width sweep
    X, y = generate_data(n_samples=200, noise=0.20, seed=4824)

    raw_path = FIGS_DIR / "moons_raw.png"
    save_dataset_scatter(X, y, "make_moons (n=200, noise=0.20)", raw_path)
    print("✓ Saved:", raw_path)

    saved = []
    print("[three_layer_neural_network] Generating decision boundary figures...")

    # Activation sweep with H=3
    for act in ["tanh", "sigmoid", "relu"]:
        model = NeuralNetwork(
            nn_input_dim=2,
            nn_hidden_dim=3,
            nn_output_dim=2,
            actFun_type=act,
            reg_lambda=0.0,
            seed=4824,
        )
        model.fit_model(X, y, epsilon=0.01, num_passes=5000, print_loss=False)
        savepath = FIGS_DIR / f"decision_boundary_{act}_H3.png"
        saved.append(str(savepath))
        model.visualize_decision_boundary(
            X,
            y,
            title=f"Decision boundary — {act}, H=3",
            savepath=savepath,
        )

    # Width sweep (tanh): H ∈ {3,10,50}
    for H in [3, 10, 50]:
        model = NeuralNetwork(
            nn_input_dim=2,
            nn_hidden_dim=H,
            nn_output_dim=2,
            actFun_type="tanh",
            reg_lambda=0.0,
            seed=4824,
        )
        model.fit_model(X, y, epsilon=0.01, num_passes=5000, print_loss=False)
        savepath = FIGS_DIR / f"decision_boundary_tanh_H{H}.png"
        saved.append(str(savepath))
        model.visualize_decision_boundary(
            X,
            y,
            title=f"Decision boundary — tanh, H={H}",
            savepath=savepath,
        )

    print(f"[three_layer_neural_network] Saved {len(saved)} figure(s) to {FIGS_DIR}")
    for p in saved:
        print(" -", p)


if __name__ == "__main__":
    main()
