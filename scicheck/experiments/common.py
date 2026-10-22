from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Iterable

import torch

from ..config import Config
from ..utils import write_json

# Keys that may change between invocations without invalidating finished results: where things
# run, and the grid axes (adding seeds or sizes later just adds units).
_FREE_KEYS = {"device", "out_dir", "data_dir", "name", "architectures", "sizes", "severities", "seeds"}


def check_or_write_config(cfg: Config, out: Path, filename: str = "config.json") -> None:
    """Refuse to mix results produced under different settings in one output directory."""
    path = out / filename
    strip = lambda d: {k: v for k, v in d.items() if k not in _FREE_KEYS}
    current = strip(json.loads(json.dumps(cfg.to_dict())))  # tuples -> lists, like the stored file
    if path.exists():
        stored = strip(json.loads(path.read_text()))
        if stored != current:
            diff = sorted(k for k in set(stored) | set(current) if stored.get(k) != current.get(k))
            raise SystemExit(
                f"{path} was written with different settings (differs in: {diff}).\n"
                f"Use a different out_dir or delete {out} to start over."
            )
    write_json(path, cfg.to_dict())


def _worker_device(devices: list[str]) -> torch.device:
    ident = mp.current_process()._identity
    slot = (ident[0] - 1) if ident else 0
    return torch.device(devices[slot % len(devices)])


def _init_worker(threads: int | None) -> None:
    if threads:
        torch.set_num_threads(threads)


def run_jobs(fn: Callable, jobs: Iterable[tuple], workers: int, devices: list[str], threads: int | None):
    """Run ``fn(*job, device)`` for every job, in-process or on a spawn pool. Each pool worker
    is pinned to one entry of ``devices`` (round-robin over worker slots)."""
    jobs = list(jobs)
    if workers <= 1:
        _init_worker(threads)
        for job in jobs:
            yield fn(*job, torch.device(devices[0]))
        return
    ctx = mp.get_context("spawn")  # CUDA cannot be re-initialized in forked children
    with ctx.Pool(workers, initializer=_init_worker, initargs=(threads,), maxtasksperchild=None) as pool:
        yield from pool.imap_unordered(_Call(fn, devices), jobs)


class _Call:
    def __init__(self, fn, devices):
        self.fn, self.devices = fn, devices

    def __call__(self, job):
        return self.fn(*job, _worker_device(self.devices))
