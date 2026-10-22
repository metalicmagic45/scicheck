"""§V-A snapshot representation: 13 + L statistics recorded once per training batch.

Per batch the recorder only copies the flat gradient g_t, flat weights w_t and loss_t into a
preallocated on-device chunk buffer; the statistics are then computed for a whole chunk of
batches at once. The values are identical to computing them batch by batch, but the per-batch
cost drops from ~50 small kernels to 3 copies, and the host never syncs with the device.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Order matters: it is the column order of every snapshot sequence (Table II).
BASE_FEATURES = [
    # seven phase-invariant dynamics statistics
    "grad_snr",           # |ḡ_t| / σ_t
    "loss_delta",         # loss_t - loss_{t-1}
    "eta",                # η_t = ḡ_t^2 / (ḡ_t^2 + σ_t^2)
    "grad_norm",          # ||g_t||_2
    "weight_delta_norm",  # ||w_t - w_{t-1}||_2
    "loss",               # loss_t
    "weight_norm",        # ||w_t||_2
    # six-statistic gradient-shape block
    "frac_pos",           # Pr[g_{t,i} > 0]
    "absg_q10", "absg_q50", "absg_q90",  # percentiles of |g_{t,i}|
    "grad_skew",          # E[z^3]
    "grad_kurt",          # E[z^4] - 3
]
ETA_COL = BASE_FEATURES.index("eta")
_EPS = 1e-12
_MAX_QUANTILE_NUMEL = 2**24  # torch.quantile input limit


def feature_names(n_layers: int) -> list[str]:
    return BASE_FEATURES + [f"layer{i}_grad_norm" for i in range(n_layers)]


class SnapshotRecorder:
    """Call ``record(loss)`` once per batch, after ``optimizer.step()``.

    At that point ``p.grad`` still holds the gradient the optimizer just applied (i.e. *after*
    any tampering) and the parameters hold w_t, so the weight delta is w_t - w_{t-1}.
    """

    def __init__(self, model: nn.Module, tracked: list[nn.Module], chunk: int = 256):
        self.params = [p for p in model.parameters() if p.requires_grad]
        device = self.params[0].device
        n = sum(p.numel() for p in self.params)

        # Each tracked layer's parameters occupy one contiguous slice of the flat vector.
        offset, start_of = 0, {}
        for p in self.params:
            start_of[id(p)] = (offset, offset + p.numel())
            offset += p.numel()
        self.layer_slices = []
        for m in tracked:
            spans = sorted(start_of[id(p)] for p in m.parameters() if p.requires_grad)
            if any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError(f"parameters of {m} are not contiguous in model.parameters()")
            self.layer_slices.append((spans[0][0], spans[-1][1]))

        self.chunk = max(1, min(chunk, _MAX_QUANTILE_NUMEL // n))
        self.gbuf = torch.empty(self.chunk, n, device=device)
        self.wbuf = torch.empty(self.chunk, n, device=device)
        self.lbuf = torch.empty(self.chunk, device=device)
        self.k = 0
        self.q = torch.tensor([0.1, 0.5, 0.9], device=device)
        with torch.no_grad():
            self.prev_w = torch.cat([p.reshape(-1) for p in self.params]).float()
        self.prev_loss: torch.Tensor | None = None
        self.out: list[torch.Tensor] = []

    @torch.no_grad()
    def record(self, loss: torch.Tensor) -> None:
        torch.cat([p.grad.reshape(-1) for p in self.params], out=self.gbuf[self.k])
        torch.cat([p.reshape(-1) for p in self.params], out=self.wbuf[self.k])
        self.lbuf[self.k] = loss.detach()
        self.k += 1
        if self.k == self.chunk:
            self._flush()

    @torch.no_grad()
    def _flush(self) -> None:
        k = self.k
        if k == 0:
            return
        G, W, L = self.gbuf[:k], self.wbuf[:k], self.lbuf[:k]

        var, mean = torch.var_mean(G, dim=1, correction=0)
        std = var.sqrt()
        C = G - mean[:, None]
        C2 = C * C
        prev_w = torch.cat([self.prev_w[None], W[:-1]])
        first_loss = L[:1] if self.prev_loss is None else self.prev_loss.reshape(1)
        prev_l = torch.cat([first_loss, L[:-1]])  # loss_delta of the very first batch is 0
        qs = torch.quantile(G.abs(), self.q, dim=1)

        cols = [
            mean.abs() / (std + _EPS),
            L - prev_l,
            mean**2 / (mean**2 + var + _EPS),
            G.norm(dim=1),
            (W - prev_w).norm(dim=1),
            L,
            W.norm(dim=1),
            (G > 0).float().mean(1),
            qs[0], qs[1], qs[2],
            (C2 * C).mean(1) / (var * std + _EPS),
            (C2 * C2).mean(1) / (var * var + _EPS) - 3.0,
            *[G[:, a:b].norm(dim=1) for a, b in self.layer_slices],
        ]
        self.out.append(torch.stack(cols, dim=1))
        self.prev_w = W[-1].clone()
        self.prev_loss = L[-1].clone()
        self.k = 0

    def to_numpy(self) -> np.ndarray:
        self._flush()
        return torch.cat(self.out).cpu().numpy().astype(np.float32)
