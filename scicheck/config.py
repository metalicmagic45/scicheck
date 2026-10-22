"""Experiment configuration. Defaults are the settings reported in §VI of the paper;
values the paper does not state are marked NOT STATED and are our choices."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    epochs: int = 25            # §VI-A
    batch_size: int = 32        # NOT STATED
    lr: float = 1e-3            # NOT STATED (Adam default); optimizer is Adam per §VI-A


@dataclass
class JudgeConfig:
    window: int = 30            # W
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    e_layers: int = 2
    dropout: float = 0.0        # NOT STATED (Xu et al. reference implementation default)
    lam: float = 3.0            # association-discrepancy weight λ
    epochs: int = 15
    patience: int = 5
    lr: float = 1e-4
    batch_size: int = 32
    temperature: float = 50.0   # NOT STATED (Xu et al. reference implementation)
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    r: float = 0.01             # τ = (1 - r)-quantile of the validation timeline
    # Alg. 1 line 9 takes η̄_v over all of T; §VI-A says the baseline is computed only from
    # T_train. The experiments section describes what was run, so "train" is the default.
    eta_baseline: str = "train"  # "train" | "all"
    score_batch_size: int = 1024  # inference batch size (no effect on results)


@dataclass
class ValidateConfig:
    theta: float = 0.3          # flagged-window fraction threshold θ
    rho: float = 0.4            # effective-signal ratio ρ, γ = ρ·η̄_v


@dataclass
class PerformanceConfig:
    n_honest: int = 10
    n_malicious: int = 10
    severity: float = 0.3       # NOT STATED: flip used by the 10 malicious runs in §VI-B
    sweeps: tuple[str, ...] = ("runtime", "memory")


@dataclass
class Config:
    name: str = "paper"
    out_dir: str = "results/paper"
    data_dir: str = "data"
    device: str = "auto"         # auto -> cuda if available; CLI --devices overrides per worker
    deterministic: bool = False  # force deterministic cuDNN kernels (slower on GPU)
    architectures: list[str] = field(default_factory=lambda: ["mlp", "cnn", "ae"])
    sizes: list[int] = field(default_factory=lambda: [4000, 6000, 8000, 10000, 12000])
    severities: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_honest: int = 10
    n_malicious: int = 10
    train_frac: float = 0.3     # verifying scientist trains on the first 30% of the data
    data_seed: int = 0          # fixes the CNN/AE subsample once for all seeds
    save_snapshots: bool = True  # keep every run's snapshot sequence for post-hoc analysis
    training: TrainingConfig = field(default_factory=TrainingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    validate: ValidateConfig = field(default_factory=ValidateConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {
    "training": TrainingConfig,
    "judge": JudgeConfig,
    "validate": ValidateConfig,
    "performance": PerformanceConfig,
}


def _build(cls, values: dict, where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {sorted(unknown)}")
    kwargs = {}
    for k, v in values.items():
        if k in _SECTIONS and cls is Config:
            v = _build(_SECTIONS[k], v or {}, f"{where}.{k}")
        elif isinstance(v, list) and k in ("split", "sweeps"):
            v = tuple(v)
        kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw, str(path))


def config_from_dict(d: dict) -> Config:
    return _build(Config, d, "<dict>")
