from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import JudgeConfig
from .judge import Judge

OCSVM_MAX_TRAIN = 5000


@dataclass
class Baseline:
    name: str
    model: object
    tau: float

    def scores(self, T_norm: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(T_norm)

    def phi(self, judge: Judge, Ts: np.ndarray) -> float:
        T = judge.normalize(Ts)
        ok = np.isfinite(T).all(1)
        flagged = ~ok
        if ok.any():
            flagged[ok] = self.scores(T[ok]) > self.tau
        return float(flagged.mean())


def _make(name: str, seed: int):
    if name == "iforest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(n_estimators=200, random_state=seed)
    if name == "ocsvm":
        from sklearn.svm import OneClassSVM

        return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    raise ValueError(f"unknown baseline {name!r}")


def fit_baselines(T: np.ndarray, judge: Judge, cfg: JudgeConfig, seed: int) -> dict[str, Baseline]:
    n_tr, n_val = int(len(T) * cfg.split[0]), int(len(T) * cfg.split[1])
    T_tr, T_val = judge.normalize(T[:n_tr]), judge.normalize(T[n_tr:n_tr + n_val])
    out = {}
    for name in cfg.baselines:
        X = T_tr
        if name == "ocsvm" and len(X) > OCSVM_MAX_TRAIN:
            X = X[np.random.default_rng(seed).choice(len(X), OCSVM_MAX_TRAIN, replace=False)]
        model = _make(name, seed).fit(X)
        b = Baseline(name, model, 0.0)
        b.tau = float(np.quantile(b.scores(T_val), 1.0 - cfg.r))
        out[name] = b
    return out
