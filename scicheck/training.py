"""One training run under a blueprint b̂, recording one snapshot per batch (§IV phases a/b)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .attacks import SignFlipAttack
from .config import TrainingConfig
from .data import to_tensors
from .models import ARCHS, count_params, tracked_layers
from .snapshots import SnapshotRecorder
from .utils import Timer, derive_seed


@dataclass
class RunResult:
    snapshots: np.ndarray    # (n_batches_total, 13 + L)
    final_loss: float        # mean batch loss over the last epoch
    seconds: float
    n_params: int


def train_run(
    arch: str,
    width: int,
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainingConfig,
    seed: int,
    device: torch.device,
    flip: float = 0.0,
) -> RunResult:
    """Train ``arch`` at ``width`` on (X, y) for ``cfg.epochs`` epochs of Adam.

    ``seed`` controls model initialization, minibatch order and (if ``flip > 0``) the
    sign-flip masks -- the only things that vary across repeats in §VI-A.
    """
    spec = ARCHS[arch]
    torch.manual_seed(derive_seed(seed, "init"))
    model = spec.build(width).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, foreach=True)  # same update; fewer kernels
    loss_fn = nn.BCEWithLogitsLoss() if spec.task == "binary" else nn.MSELoss()
    attack = SignFlipAttack(flip, derive_seed(seed, "attack"), device) if flip > 0 else None
    batch_gen = torch.Generator().manual_seed(derive_seed(seed, "batches"))

    Xt, yt = to_tensors(X, y, device)
    n, bs = len(Xt), cfg.batch_size
    recorder = SnapshotRecorder(model, tracked_layers(model))
    epoch_losses: list[torch.Tensor] = []

    with Timer(device) as t:
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, generator=batch_gen).to(device)
            epoch_losses = []
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                out = model(Xt[idx])
                if spec.task == "binary":
                    out = out.squeeze(-1)
                loss = loss_fn(out, yt[idx])
                opt.zero_grad(set_to_none=False)
                loss.backward()
                if attack is not None:
                    attack(params)
                opt.step()
                recorder.record(loss)
                epoch_losses.append(loss.detach())
        snapshots = recorder.to_numpy()

    return RunResult(
        snapshots=snapshots,
        final_loss=float(torch.stack(epoch_losses).mean()),
        seconds=t.seconds,
        n_params=count_params(model),
    )
