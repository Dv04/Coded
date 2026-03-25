from __future__ import annotations

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', DEVICE)

class LSTMCellScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        self.forget_linear = nn.Linear(concat_size, hidden_size)
        self.input_linear = nn.Linear(concat_size, hidden_size)
        self.output_linear = nn.Linear(concat_size, hidden_size)
        self.candidate_linear = nn.Linear(concat_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([h_prev, x_t], dim=1)

        f_t = torch.sigmoid(self.forget_linear(combined))
        i_t = torch.sigmoid(self.input_linear(combined))
        o_t = torch.sigmoid(self.output_linear(combined))
        g_t = torch.tanh(self.candidate_linear(combined))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTMLanguageModelScratch(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell = LSTMCellScratch(embedding_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        batch_size, seq_len = x.shape
        embeddings = self.embedding(x)

        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        logits = []
        c_states = []

        for t in range(seq_len):
            x_t = embeddings[:, t, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            if return_states:
                c_t.retain_grad()
                c_states.append(c_t)
            logits.append(self.output_layer(h_t).unsqueeze(1))

        logits = torch.cat(logits, dim=1)
        if return_states:
            return logits, c_states
        return logits



def sample_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    tokens = torch.randint(low=1, high=vocab_size, size=(batch_size, seq_len), device=device)
    return tokens, tokens.clone()


def evaluate(model: nn.Module, batches: int, batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_batch(batch_size, seq_len, vocab_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_count += y.numel()
    avg_loss = total_loss / batches
    ppl = math.exp(avg_loss)
    acc = total_correct / total_count
    return avg_loss, ppl, acc


vocab_size = 25
embedding_dim = 32
hidden_size = 64
seq_len = 20
batch_size = 128
num_epochs = 30
steps_per_epoch = 80
learning_rate = 3e-3

model = LSTMLanguageModelScratch(vocab_size, embedding_dim, hidden_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for _ in range(steps_per_epoch):
        x, y = sample_batch(batch_size, seq_len, vocab_size, DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / steps_per_epoch
    loss_history.append(avg_train_loss)

    if epoch % 5 == 0 or epoch == 1:
        train_ppl = math.exp(avg_train_loss)
        print(f"epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | train_ppl={train_ppl:.3f}")

val_loss, val_ppl, val_acc = evaluate(model, batches=30, batch_size=128, seq_len=seq_len, vocab_size=vocab_size, device=DEVICE)
print('val_loss', val_loss)
print('val_ppl', val_ppl)
print('val_acc', val_acc)

# Gradient norm of cell state over time
model.zero_grad(set_to_none=True)
x_probe, y_probe = sample_batch(128, seq_len, vocab_size, DEVICE)
probe_logits, c_states = model(x_probe, return_states=True)
probe_loss = F.cross_entropy(probe_logits.reshape(-1, vocab_size), y_probe.reshape(-1))
probe_loss.backward()

grad_norms = [c.grad.norm(dim=1).mean().item() for c in c_states]
print('grad_norms_first5', grad_norms[:5])
print('grad_norms_last5', grad_norms[-5:])
