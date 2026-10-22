from __future__ import annotations

import numpy as np
import torch

ATTACKS = ("signflip", "data", "fabricate")


class SignFlipAttack:
    def __init__(self, flip: float, seed: int, device: torch.device):
        if not 0.0 < flip <= 1.0:
            raise ValueError(f"flip must be in (0, 1], got {flip}")
        self.flip = flip
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def __call__(self, params) -> None:
        grads = [p.grad for p in params if p.grad is not None]
        sizes = [g.numel() for g in grads]
        u = torch.rand(sum(sizes), generator=self.gen, device=self.device)
        signs = torch.where(u < self.flip, -1.0, 1.0).to(grads[0].dtype)
        torch._foreach_mul_(grads, [s.view_as(g) for s, g in zip(signs.split(sizes), grads)])


def poison_data(X: np.ndarray, y: np.ndarray, task: str, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"data attack severity must be in (0, 1), got {frac}")
    idx = np.random.default_rng(seed).choice(len(X), size=int(round(frac * len(X))), replace=False)
    X = X.copy()
    if task == "binary":
        y = y.copy()
        y[idx] = 1.0 - y[idx]
    else:
        X[idx] = 1.0 - X[idx]
        y = X
    return X, y


def fabricate_tail(snapshots: np.ndarray, n_total: int, steps_per_epoch: int) -> np.ndarray:
    tail = snapshots[-steps_per_epoch:]
    missing = n_total - len(snapshots)
    reps = -(-missing // len(tail))
    return np.concatenate([snapshots, np.tile(tail, (reps, 1))[:missing]])
