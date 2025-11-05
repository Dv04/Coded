# Dev Sanghvi (ds221)

"""Sequence classifier (RNN/GRU/LSTM) imported by rnn_mnist.py."""

import torch.nn as nn


class RNNSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size=28,
        hidden_size=128,
        num_layers=1,
        num_classes=10,
        rnn_type="rnn",
        bidirectional=False,
        dropout=0.0,
    ):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * mult, num_classes)

    def forward(self, x):
        if self.rnn_type == "lstm":
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)
