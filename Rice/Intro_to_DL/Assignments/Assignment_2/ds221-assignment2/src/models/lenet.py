# Dev Sanghvi (ds221)

"""LeNet-5 definition imported by cnn_cifar10_lenet.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, activation: str = "tanh"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, num_classes)
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, return_activations: bool = False):
        a1 = self.act(self.conv1(x))
        p1 = self.pool(a1)
        a2 = self.act(self.conv2(p1))
        p2 = self.pool(a2)
        a3 = self.act(self.conv3(p2))
        p3 = self.pool(a3)
        flat = p3.view(p3.size(0), -1)
        a4 = self.act(self.fc1(flat))
        a5 = self.act(self.fc2(a4))
        logits = self.fc3(a5)
        if return_activations:
            return logits, {"conv1": a1, "conv2": a2, "fc1": a4, "fc2": a5}
        return logits
