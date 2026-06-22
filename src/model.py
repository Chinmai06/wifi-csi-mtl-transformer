# Multi-task Transformer for CSI sensing.
# Backbone: 4-layer Transformer encoder. Heads: activity (10), fall (2), subject (30).

import math
import torch
import torch.nn as nn
from torch.autograd import Function


class PositionalEncoding(nn.Module):
    # standard sinusoidal PE from "Attention is all you need"
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MTLTransformer(nn.Module):
    def __init__(self, input_dim=90, d_model=128, nhead=8, num_layers=4,
                 dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head_act = nn.Linear(d_model, 10)
        self.head_fall = nn.Linear(d_model, 2)
        self.head_subj = nn.Linear(d_model, 30)

    def features(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        return h.mean(dim=1)  # mean pool over time

    def forward(self, x):
        h = self.features(x)
        return {
            'activity': self.head_act(h),
            'fall': self.head_fall(h),
            'subject': self.head_subj(h),
            'features': h,
        }


# Gradient reversal layer for DANN-style domain adaptation
# https://arxiv.org/abs/1409.7495
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GRLTransformer(MTLTransformer):
    # Same as MTLTransformer but subject head sees reversed gradients,
    # so backbone is pushed toward subject-invariant features.
    def forward(self, x, lam=1.0):
        h = self.features(x)
        h_rev = GradReverse.apply(h, lam)
        return {
            'activity': self.head_act(h),
            'fall': self.head_fall(h),
            'subject': self.head_subj(h_rev),
            'features': h,
        }


class SingleTaskTransformer(nn.Module):
    # For single-task baselines (same backbone, one head)
    def __init__(self, task, input_dim=90, d_model=128, nhead=8, num_layers=4,
                 dim_ff=256, dropout=0.1):
        super().__init__()
        n_out = {'activity': 10, 'fall': 2, 'subject': 30}[task]
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h)


# Uncertainty-weighted MTL loss from Kendall et al. 2018
# learns log-variance per task, weights = exp(-log_var)
class UncertaintyLoss(nn.Module):
    def __init__(self, n_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0
        for i, L in enumerate(losses):
            total = total + torch.exp(-self.log_vars[i]) * L + self.log_vars[i]
        return total
