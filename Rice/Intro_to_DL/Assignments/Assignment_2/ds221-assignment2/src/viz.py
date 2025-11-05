# Dev Sanghvi (ds221)

"""Visualization helpers (curves/boxplots/occlusion/confusion) shared by both training scripts."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch


def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_curves(history, out_prefix):
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if "train_loss" in history:
        plt.figure()
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        save_fig(out_prefix + "_loss.png")
    if "train_acc" in history or "test_acc" in history:
        plt.figure()
        if "train_acc" in history:
            plt.plot(epochs, history["train_acc"], label="Train Acc")
        if "test_acc" in history:
            plt.plot(epochs, history["test_acc"], label="Test Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        save_fig(out_prefix + "_acc.png")


def visualize_conv1_filters(conv1_weight, out_path):
    w = conv1_weight.detach().cpu().numpy()
    k = w.shape[0]
    cols = 6
    rows = int(np.ceil(k / cols))
    plt.figure(figsize=(cols * 1.2, rows * 1.2))
    for i in range(k):
        plt.subplot(rows, cols, i + 1)
        if w.shape[1] == 3:
            filt = np.transpose(w[i], (1, 2, 0))
            fmin, fmax = filt.min(), filt.max()
            if fmax > fmin:
                filt = (filt - fmin) / (fmax - fmin)
            plt.imshow(filt)
        else:
            plt.imshow(w[i, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle("First Conv Layer Filters")
    save_fig(out_path)


def plot_activation_stats(activations, out_prefix):
    for name, act in activations.items():
        a = act.detach().cpu().float().reshape(-1).numpy()
        plt.figure()
        plt.hist(a, bins=50)
        plt.title(f"Activation Histogram: {name}")
        plt.xlabel("Activation")
        plt.ylabel("Count")
        save_fig(out_prefix + f"_hist_{name}.png")


def plot_confusion(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    save_fig(out_path)


def collect_conv_activations(model, dataloader, layers=None, max_batches=10, device=None):
    if layers is None:
        layers = [getattr(model, "conv1", None), getattr(model, "conv2", None)]
    layer_objs = [L for L in layers if L is not None]

    def name_of(layer):
        for k, v in model._modules.items():
            if v is layer:
                return k
        return "conv"

    names = [name_of(L) for L in layer_objs]
    buffers = {n: [] for n in names}
    hooks = []

    def make_hook(nm):
        def _hook(_m, _inp, out):
            with torch.no_grad():
                B, C, H, W = out.shape
                vals = out.detach().reshape(B, C, -1).permute(1, 0, 2).reshape(C, -1)
                buffers[nm].append(vals.cpu())

        return _hook

    for nm, layer in zip(names, layer_objs):
        hooks.append(layer.register_forward_hook(make_hook(nm)))

    model.eval()
    count = 0
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device) if device else xb
            model(xb)
            count += 1
            if max_batches is not None and count >= max_batches:
                break

    for h in hooks:
        h.remove()

    per_layer = {}
    for nm, chunks in buffers.items():
        if not chunks:
            continue
        mat = torch.cat(chunks, dim=1)
        per_layer[nm] = [mat[i].numpy() for i in range(mat.shape[0])]
    return per_layer


def plot_activation_boxplots(per_layer_filter_values, outdir):
    os.makedirs(outdir, exist_ok=True)
    for layer, per_filters in per_layer_filter_values.items():
        plt.figure(figsize=(10, 4))
        plt.boxplot(per_filters, showfliers=False)
        plt.xlabel("Filter index")
        plt.ylabel("Activation")
        plt.title(f"Activation distribution per filter — {layer}")
        save_fig(os.path.join(outdir, f"cnn_boxplot_{layer}.png"))


def occlusion_map(model, image, label_idx, patch=5, stride=2, device=None):
    model.eval()
    x = image.to(device) if device else image
    with torch.no_grad():
        base_logits = model(x)
        base = base_logits[0, int(label_idx)].item()
    _, _, H, W = x.shape
    grid_h = (H - patch) // stride + 1
    grid_w = (W - patch) // stride + 1
    heat = torch.zeros((grid_h, grid_w))
    with torch.no_grad():
        rr = 0
        for i in range(0, H - patch + 1, stride):
            cc = 0
            for j in range(0, W - patch + 1, stride):
                x_occ = x.clone()
                x_occ[:, :, i : i + patch, j : j + patch] = 0.5
                logits = model(x_occ)
                drop = base - logits[0, int(label_idx)].item()
                heat[rr, cc] = drop
                cc += 1
            rr += 1
    hmin, hmax = float(heat.min().item()), float(heat.max().item())
    if hmax > hmin:
        heat = (heat - hmin) / (hmax - hmin)
    heat_up = torch.nn.functional.interpolate(
        heat[None, None], size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    return heat_up


def save_occlusion_grid(model, dataset, idx_list, outpath, device=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    cols = 2
    rows = len(idx_list)
    plt.figure(figsize=(cols * 3, rows * 3))
    for r, idx in enumerate(idx_list):
        img, label = dataset[idx]
        if hasattr(label, "ndim") and getattr(label, "ndim", 0) == 1 and label.numel() == 10:
            label_idx = int(label.argmax().item())
        else:
            try:
                label_idx = int(label.item())
            except Exception:
                label_idx = int(label)
        if torch.is_tensor(img):
            img_cpu = img.detach().cpu()
            if img_cpu.ndim == 3 and img_cpu.shape[0] in (1, 3):
                img_np = img_cpu.permute(1, 2, 0).numpy()
            elif img_cpu.ndim == 2:
                img_np = img_cpu.numpy()
            else:
                img_np = img_cpu.squeeze().numpy()
            if img_cpu.ndim == 3:
                img_t = img_cpu.unsqueeze(0)
            elif img_cpu.ndim == 2:
                img_t = img_cpu.unsqueeze(0).unsqueeze(0)
            else:
                img_t = img_cpu.unsqueeze(0)
        else:
            arr = np.asarray(img)
            img_np = arr if arr.ndim == 3 else arr.squeeze()
            img_t = torch.tensor(arr)
            if img_t.ndim == 3 and img_t.shape[0] not in (1, 3):
                img_t = img_t.permute(2, 0, 1)
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)
            elif img_t.ndim == 2:
                img_t = img_t.unsqueeze(0).unsqueeze(0)
        heat = occlusion_map(model, img_t, label_idx, device=device)
        ax = plt.subplot(rows, cols, r * cols + 1)
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
            ax.imshow(img_np.squeeze(), cmap="gray")
        else:
            arr = img_np
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            ax.imshow(arr)
        ax.set_title(f"idx={idx} label={label_idx}")
        ax.axis("off")
        ax = plt.subplot(rows, cols, r * cols + 2)
        ax.imshow(heat, cmap="jet")
        ax.axis("off")
    save_fig(outpath)
