from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import ATTACKS, SignFlipAttack, fabricate_tail, poison_data
from .config import TrainingConfig
from .data import to_tensors
from .models import ARCHS, count_params, tracked_layers
from .snapshots import SnapshotRecorder
from .utils import Timer, derive_seed

AUG_PAD = 4


@dataclass
class RunResult:
    snapshots: np.ndarray
    final_loss: float
    seconds: float
    n_params: int
    history: list[dict] = field(default_factory=list)


def augment_images(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    b, c, h, w = x.shape
    dev = x.device
    flip = (torch.rand(b, generator=gen) < 0.5).to(dev)
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (AUG_PAD,) * 4, mode="reflect")
    di = torch.randint(0, 2 * AUG_PAD + 1, (b,), generator=gen).to(dev)
    dj = torch.randint(0, 2 * AUG_PAD + 1, (b,), generator=gen).to(dev)
    rows = (di[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (dj[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    return xp[torch.arange(b, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None],
              rows, cols]


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, task: str, loss_fn: nn.Module,
             bs: int = 1024) -> dict:
    total, correct = 0.0, 0.0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        if task == "binary":
            out = out.squeeze(-1)
            correct += float(((out > 0).float() == y[i:i + bs]).sum())
        total += float(loss_fn(out, y[i:i + bs])) * len(out)
    res = {"eval_loss": total / len(X)}
    if task == "binary":
        res["eval_acc"] = correct / len(X)
    return res


def verifier_steps(mode: str, cfg: TrainingConfig, n_scientist_rows: int) -> int | None:
    if mode == "epochs":
        return None
    if mode == "match":
        return cfg.epochs * -(-n_scientist_rows // cfg.batch_size)
    raise ValueError(f"verifier_steps must be 'match' or 'epochs', got {mode!r}")


def train_run(
    arch: str,
    width: int,
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainingConfig,
    seed: int,
    device: torch.device,
    attack: str | None = None,
    severity: float = 0.0,
    eval_data: tuple[np.ndarray, np.ndarray] | None = None,
    steps: int | None = None,
) -> RunResult:
    if attack is not None and attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}, expected one of {ATTACKS}")
    spec = ARCHS[arch]
    if attack == "data":
        X, y = poison_data(X, y, spec.task, severity, derive_seed(seed, "poison"))
    torch.manual_seed(derive_seed(seed, "init"))
    model = spec.build(width).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, foreach=True)
    loss_fn = nn.BCEWithLogitsLoss() if spec.task == "binary" else nn.MSELoss()
    flipper = SignFlipAttack(severity, derive_seed(seed, "attack"), device) if attack == "signflip" else None
    batch_gen = torch.Generator().manual_seed(derive_seed(seed, "batches"))

    Xt, yt = to_tensors(X, y, device)
    augment = cfg.augment and Xt.dim() == 4
    aug_gen = torch.Generator().manual_seed(derive_seed(seed, "augment")) if augment else None
    Xe, ye = to_tensors(*eval_data, device) if eval_data is not None else (None, None)
    n, bs = len(Xt), cfg.batch_size
    steps_per_epoch = -(-n // bs)
    n_steps = cfg.epochs * steps_per_epoch if steps is None else steps
    n_epochs = -(-n_steps // steps_per_epoch)
    stop = int(round((1.0 - severity) * n_steps)) if attack == "fabricate" else n_steps
    if attack == "fabricate" and not steps_per_epoch <= stop < n_steps:
        raise ValueError(f"fabricate severity {severity} must leave at least one genuine epoch")
    step = 0
    recorder = SnapshotRecorder(model, tracked_layers(model))
    epoch_losses: list[torch.Tensor] = []
    history: list[dict] = []
    eval_seconds = 0.0

    with Timer(device) as t:
        for epoch in range(n_epochs):
            if step >= stop:
                break
            perm = torch.randperm(n, generator=batch_gen).to(device)
            epoch_losses = []
            for i in range(0, n, bs):
                if step >= stop:
                    break
                idx = perm[i:i + bs]
                xb = augment_images(Xt[idx], aug_gen) if augment else Xt[idx]
                out = model(xb)
                if spec.task == "binary":
                    out = out.squeeze(-1)
                loss = loss_fn(out, yt[idx])
                opt.zero_grad(set_to_none=False)
                loss.backward()
                if flipper is not None:
                    flipper(params)
                opt.step()
                recorder.record(loss)
                epoch_losses.append(loss.detach())
                step += 1
            if Xe is not None and len(epoch_losses) == steps_per_epoch:
                with Timer(device) as te:
                    history.append({"epoch": epoch + 1, "train_loss": float(torch.stack(epoch_losses).mean()),
                                    **evaluate(model, Xe, ye, spec.task, loss_fn)})
                eval_seconds += te.seconds
        snapshots = recorder.to_numpy()
    if attack == "fabricate":
        snapshots = fabricate_tail(snapshots, n_steps, steps_per_epoch)

    return RunResult(
        snapshots=snapshots,
        final_loss=float(torch.stack(epoch_losses).mean()),
        seconds=t.seconds - eval_seconds,
        n_params=count_params(model),
        history=history,
    )
