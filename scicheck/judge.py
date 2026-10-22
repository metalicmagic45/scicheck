from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .anomaly_transformer import AnomalyTransformer, anomaly_scores, minimax_losses
from .config import JudgeConfig
from .snapshots import ETA_COL


@dataclass
class Judge:
    model: AnomalyTransformer
    mu: np.ndarray
    sd: np.ndarray
    tau: float
    eta_v: float
    gamma: float
    cfg: JudgeConfig
    diagnostics: dict = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def normalize(self, T: np.ndarray) -> np.ndarray:
        return ((T - self.mu) / self.sd).astype(np.float32)

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self.model.state_dict(), "mu": self.mu, "sd": self.sd, "tau": self.tau,
            "eta_v": self.eta_v, "gamma": self.gamma, "cfg": self.cfg.__dict__,
            "c_in": len(self.mu), "diagnostics": self.diagnostics,
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "Judge":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = JudgeConfig(**ck["cfg"])
        model = _build_model(cfg, ck["c_in"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return cls(model, ck["mu"], ck["sd"], ck["tau"], ck["eta_v"], ck["gamma"], cfg, ck["diagnostics"])


@dataclass
class Verdict:
    accept: bool
    phi: float
    eta_s: float
    at_pass: bool
    eta_pass: bool


def _build_model(cfg: JudgeConfig, c_in: int) -> AnomalyTransformer:
    return AnomalyTransformer(cfg.window, c_in, cfg.d_model, cfg.n_heads, cfg.e_layers, cfg.d_ff, cfg.dropout)


def sliding_windows(T: np.ndarray, W: int) -> np.ndarray:
    if len(T) < W:
        raise ValueError(f"sequence of length {len(T)} is shorter than the window W={W}")
    return np.lib.stride_tricks.sliding_window_view(T, W, axis=0).transpose(0, 2, 1)


def windows_to_timeline(scores: np.ndarray, n: int) -> np.ndarray:
    n_win, W = scores.shape
    sums = np.zeros(n)
    counts = np.zeros(n)
    for k in range(W):
        sums[k:k + n_win] += scores[:, k]
        counts[k:k + n_win] += 1
    return sums / counts


def score_timeline(model: AnomalyTransformer, T_norm: np.ndarray, cfg: JudgeConfig) -> np.ndarray:
    device = next(model.parameters()).device
    win = sliding_windows(T_norm, cfg.window)
    model.eval()
    out = []
    for i in range(0, len(win), cfg.score_batch_size):
        x = torch.as_tensor(np.ascontiguousarray(win[i:i + cfg.score_batch_size]), device=device)
        out.append(anomaly_scores(model, x, cfg.temperature).cpu())
    return windows_to_timeline(torch.cat(out).numpy(), len(T_norm))


@torch.no_grad()
def _eval_losses(model, windows: torch.Tensor, cfg: JudgeConfig) -> tuple[float, float]:
    model.eval()
    l1 = l2 = 0.0
    for i in range(0, len(windows), cfg.score_batch_size):
        x = windows[i:i + cfg.score_batch_size]
        out, series, prior = model(x)
        rec = torch.mean((out - x) ** 2)
        s_loss, p_loss = minimax_losses(series, prior)
        l1 += float(rec - cfg.lam * s_loss) * len(x)
        l2 += float(rec + cfg.lam * p_loss) * len(x)
    return l1 / len(windows), l2 / len(windows)


def train_judge(T: np.ndarray, cfg: JudgeConfig, rho: float, device: torch.device, seed: int) -> Judge:
    N, D = T.shape
    n_tr, n_val = int(N * cfg.split[0]), int(N * cfg.split[1])
    T_tr, T_val, T_te = T[:n_tr], T[n_tr:n_tr + n_val], T[n_tr + n_val:]

    mu = T_tr.mean(0)
    sd = T_tr.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    norm = lambda a: ((a - mu) / sd).astype(np.float32)

    torch.manual_seed(seed)
    model = _build_model(cfg, D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_w = torch.as_tensor(np.ascontiguousarray(sliding_windows(norm(T_tr), cfg.window)), device=device)
    val_w = torch.as_tensor(np.ascontiguousarray(sliding_windows(norm(T_val), cfg.window)), device=device)
    gen = torch.Generator().manual_seed(seed)

    best1 = best2 = None
    best_state, best_epoch, bad, history = None, 0, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(train_w), generator=gen).to(device)
        for i in range(0, len(train_w), cfg.batch_size):
            x = train_w[perm[i:i + cfg.batch_size]]
            out, series, prior = model(x)
            rec = torch.mean((out - x) ** 2)
            s_loss, p_loss = minimax_losses(series, prior)
            loss1 = rec - cfg.lam * s_loss
            loss2 = rec + cfg.lam * p_loss
            opt.zero_grad()
            (loss1 + loss2).backward()
            opt.step()
        v1, v2 = _eval_losses(model, val_w, cfg)
        history.append((v1, v2))
        if best1 is None or (v1 < best1 and v2 < best2):
            best1, best2, best_epoch, bad = v1, v2, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    val_timeline = score_timeline(model, norm(T_val), cfg)
    tau = float(np.quantile(val_timeline, 1.0 - cfg.r))

    eta_src = T_tr if cfg.eta_baseline == "train" else T
    eta_v = float(eta_src[:, ETA_COL].mean())
    gamma = rho * eta_v

    diagnostics = {
        "n_snapshots": N, "n_train": len(T_tr), "n_val": len(T_val), "n_test": len(T_te),
        "epochs_run": len(history), "best_epoch": best_epoch, "val_history": history,
    }
    if len(T_te) >= cfg.window:
        diagnostics["test_flagged_frac"] = float((score_timeline(model, norm(T_te), cfg) > tau).mean())
    return Judge(model, mu, sd, tau, eta_v, gamma, cfg, diagnostics)


def validate(judge: Judge, Ts: np.ndarray, theta: float) -> Verdict:
    timeline = score_timeline(judge.model, judge.normalize(Ts), judge.cfg)
    phi = float((timeline > judge.tau).mean())
    eta_s = float(Ts[:, ETA_COL].mean())
    at_pass, eta_pass = phi <= theta, eta_s >= judge.gamma
    return Verdict(at_pass and eta_pass, phi, eta_s, at_pass, eta_pass)
