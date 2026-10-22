from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .data import HIGGS_FEATURES


def build_mlp(h: int, in_dim: int = HIGGS_FEATURES) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, h), nn.ReLU(),
        nn.Linear(h, h), nn.ReLU(),
        nn.Linear(h, 1),
    )


def build_cnn(c: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(2 * c, 1),
    )


def build_ae(h: int, in_dim: int = 784) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, h), nn.ReLU(),
        nn.Linear(h, in_dim), nn.Sigmoid(),
    )


@dataclass(frozen=True)
class ArchSpec:
    name: str
    dataset: str
    task: str
    build: Callable[[int], nn.Module]
    n_tracked: int


ARCHS: dict[str, ArchSpec] = {
    "mlp": ArchSpec("mlp", "higgs", "binary", build_mlp, 3),
    "cnn": ArchSpec("cnn", "petimages", "binary", build_cnn, 3),
    "ae": ArchSpec("ae", "mnist", "reconstruction", build_ae, 2),
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tracked_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def width_for_target(arch: str, target: int, max_width: int = 1024) -> int:
    spec = ARCHS[arch]
    best_w, best_gap = 1, float("inf")
    for w in range(1, max_width + 1):
        n = count_params(spec.build(w))
        if abs(n - target) < best_gap:
            best_w, best_gap = w, abs(n - target)
        if n > target:
            break
    return best_w


def build_for_target(arch: str, target: int) -> tuple[nn.Module, int]:
    w = width_for_target(arch, target)
    return ARCHS[arch].build(w), w
