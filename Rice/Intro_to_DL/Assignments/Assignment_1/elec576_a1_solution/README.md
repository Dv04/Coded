# Code Documentation — ELEC 576 / COMP 576 Assignment 1  
**Student:** Dev Sanghvi (`ds221`)  
**Date:** October 11, 2025  

---

## 📁 Overview
This folder contains all source code for **Assignment 1**:

- **Part 1:** Backpropagation and multilayer perceptron (MLP) experiments on synthetic datasets (*Make-Moons*, *Circles*, *Blobs*).  
- **Part 2:** Deep Convolutional Network (DCN) on the MNIST dataset implemented in PyTorch, including TensorBoard monitoring.

---

## 🧩 File Descriptions

| File                                       | Description                                                                                                                                                                                                       |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `three_layer_neural_network.py`            | Implements a simple 3-layer neural network (input–hidden–output) trained via backpropagation on the Make-Moons dataset. Includes activation functions (Tanh, Sigmoid, ReLU), loss computation, and visualization. |
| `n_layer_neural_network.py`                | Extends the above to an *n*-layer MLP with configurable layer count and width, including L2 weight regularization. Supports ReLU, Tanh, and Sigmoid activations.                                                  |
| `assignment_1_pytorch_mnist_skeleton-1.py` | PyTorch implementation of a 4-layer Deep Convolutional Network (DCN) for MNIST classification. Includes training loop, optimizer options, and TensorBoard logging.                                                |
| `utils.py` *(optional)*                    | Helper utilities (e.g., for visualization or data generation).                                                                                                                                                    |
| `__init__.py`                              | Marks this folder as a Python package.                                                                                                                                                                            |

---

## 🚀 How to Run

### 🧠 Part 1 — MLPs on Toy Datasets
```bash
# Train and visualize 3-layer network
python three_layer_neural_network.py

# Train and visualize n-layer (deeper) network
python n_layer_neural_network.py
```

Figures are saved automatically in ../report/figs/.
### 📊 Part 2 — DCN on MNIST with PyTorch
```bash
# Example run (Adam optimizer, ReLU activation, Kaiming init)
python assignment_1_pytorch_mnist_skeleton-1.py \
    --epochs 8 \
    --optimizer adam \
    --activation relu \
    --init kaiming \
    --batch-size 128

# Monitor with TensorBoard
tensorboard --logdir runs
```

---

## 🛠️ Dependencies

Install the following Python packages:
```bash
pip install numpy matplotlib scikit-learn torch torchvision tensorboard
```

---

## 🧾 Notes

- Random seeds are fixed for reproducibility.
- Softmax implementation uses a stabilized exponential (subtract max before exp).
- L2 regularization is applied in the deeper MLP model.
- All outputs and results are summarized in the main report.pdf.

---

Copyright © 2025 Dev Sanghvi - ELEC 576 / COMP 576 Assignment 1