from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3


@dataclass
class JudgeConfig:
    window: int = 30
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    e_layers: int = 2
    dropout: float = 0.0
    lam: float = 3.0
    epochs: int = 15
    patience: int = 5
    lr: float = 1e-4
    batch_size: int = 32
    temperature: float = 50.0
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    r: float = 0.01
    eta_baseline: str = "train"
    score_batch_size: int = 1024


@dataclass
class ValidateConfig:
    theta: float = 0.3
    rho: float = 0.4


@dataclass
class PerformanceConfig:
    n_honest: int = 10
    n_malicious: int = 10
    severity: float = 0.3
    sweeps: tuple[str, ...] = ("runtime", "memory")


@dataclass
class Config:
    name: str = "paper"
    out_dir: str = "results/paper"
    data_dir: str = "data"
    device: str = "auto"
    deterministic: bool = False
    architectures: list[str] = field(default_factory=lambda: ["mlp", "cnn", "ae"])
    sizes: list[int] = field(default_factory=lambda: [4000, 6000, 8000, 10000, 12000])
    severities: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_honest: int = 10
    n_malicious: int = 10
    train_frac: float = 0.3
    data_seed: int = 0
    save_snapshots: bool = True
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
