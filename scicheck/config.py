from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    augment: bool = True


@dataclass
class DataConfig:
    higgs_rows: int = 200_000
    petimages: int = 0
    mnist: int = 0
    test_frac: float = 0.15


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
    baselines: tuple[str, ...] = ("iforest", "ocsvm")


@dataclass
class ValidateConfig:
    theta: float = 0.3
    rho: float = 0.4
    replay: bool = False
    replay_tol: float = 0.02
    psi_max: float = 0.05
    spot_segments: int = 0
    spot_steps: int = 50
    spot_margin: float = 10.0
    spot_floor: float = 1e-4


@dataclass
class PerformanceConfig:
    n_honest: int = 2
    n_malicious: int = 2
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
    sizes: list[int] = field(default_factory=lambda: [100_000, 300_000, 500_000])
    attacks: list[str] = field(default_factory=lambda: ["signflip", "data", "fabricate"])
    severities: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_honest: int = 15
    n_malicious: int = 5
    train_frac: float = 0.3
    verifier_steps: str = "match"
    data_seed: int = 0
    save_snapshots: bool = True
    data: DataConfig = field(default_factory=DataConfig)
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
    "data": DataConfig,
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
        elif isinstance(v, list) and k in ("split", "sweeps", "baselines"):
            v = tuple(v)
        kwargs[k] = v
    return cls(**kwargs)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_raw(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    return _merge(_load_raw(path.parent / base), raw) if base else raw


def load_config(path: str | Path) -> Config:
    return _build(Config, _load_raw(Path(path)), str(path))


def config_from_dict(d: dict) -> Config:
    return _build(Config, d, "<dict>")
