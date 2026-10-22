from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def derive_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7FFFFFFF


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device: torch.device | None = None):
        self.device = device

    def __enter__(self):
        if self.device is not None:
            sync(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None:
            sync(self.device)
        self.seconds = time.perf_counter() - self.start


def peak_rss_mb() -> float:
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=float))
