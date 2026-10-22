from __future__ import annotations

import torch


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
