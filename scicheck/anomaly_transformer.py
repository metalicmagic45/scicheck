"""Anomaly Transformer (Xu et al., ICLR 2022), ported from github.com/thuml/Anomaly-Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float):
        super().__init__()
        self.value = TokenEmbedding(c_in, d_model)
        self.position = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value(x) + self.position(x))


class AnomalyAttention(nn.Module):
    def __init__(self, win_size: int, dropout: float):
        super().__init__()
        idx = torch.arange(win_size).float()
        self.register_buffer("distances", (idx[None, :] - idx[:, None]).abs(), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, sigma):
        B, L, H, E = q.shape
        attn = torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(E)
        series = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", series, v)

        sigma = sigma.transpose(1, 2)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).expand(-1, -1, -1, L)
        dist = self.distances[:L, :L]
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-dist**2 / 2 / sigma**2)
        return out.contiguous(), series, prior, sigma


class AttentionLayer(nn.Module):
    def __init__(self, win_size: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        d_k = d_model // n_heads
        self.n_heads = n_heads
        self.inner = AnomalyAttention(win_size, dropout)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.key = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)
        self.sigma = nn.Linear(d_model, n_heads)
        self.out = nn.Linear(d_k * n_heads, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        out, series, prior, sigma = self.inner(
            self.query(x).view(B, L, H, -1),
            self.key(x).view(B, L, H, -1),
            self.value(x).view(B, L, H, -1),
            self.sigma(x).view(B, L, H),
        )
        return self.out(out.view(B, L, -1)), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, win_size: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = AttentionLayer(win_size, d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x, series, prior, sigma = self.attention(x)
        x = self.norm1(x + self.dropout(new_x))
        y = self.dropout(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), series, prior, sigma


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size: int, c_in: int, d_model: int = 64, n_heads: int = 4,
                 e_layers: int = 2, d_ff: int = 128, dropout: float = 0.0):
        super().__init__()
        self.embedding = DataEmbedding(c_in, d_model, dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(win_size, d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_in)

    def forward(self, x):
        h = self.embedding(x)
        series, prior = [], []
        for layer in self.layers:
            h, s, p, _ = layer(h)
            series.append(s)
            prior.append(p)
        return self.projection(self.norm(h)), series, prior


def kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return res.sum(-1).mean(1)


def normalize_prior(p: torch.Tensor) -> torch.Tensor:
    return p / p.sum(-1, keepdim=True)


def minimax_losses(series, prior):
    series_loss, prior_loss = 0.0, 0.0
    for s, p in zip(series, prior):
        p = normalize_prior(p)
        series_loss = series_loss + kl_rows(s, p.detach()).mean() + kl_rows(p.detach(), s).mean()
        prior_loss = prior_loss + kl_rows(p, s.detach()).mean() + kl_rows(s.detach(), p).mean()
    return series_loss / len(prior), prior_loss / len(prior)


@torch.no_grad()
def anomaly_scores(model: AnomalyTransformer, x: torch.Tensor, temperature: float) -> torch.Tensor:
    out, series, prior = model(x)
    rec = ((x - out) ** 2).mean(-1)
    ass_dis = 0.0
    for s, p in zip(series, prior):
        p = normalize_prior(p)
        ass_dis = ass_dis + (kl_rows(s, p) + kl_rows(p, s)) * temperature
    ass_dis = ass_dis / len(prior)
    return torch.softmax(-ass_dis, dim=-1) * rec
