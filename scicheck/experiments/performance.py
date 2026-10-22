"""§VI-B performance analysis: two independent sweeps over (architecture x size x seed).

Each cell trains 21 models from scratch -- one verifying-scientist model M0 and 20 scientist
models (10 honest, 10 malicious) -- plus the judge model, which is trained on M0's snapshots
and scores all 20 submissions.

Runtime: measured in-process with a monotonic wall-clock timer on the configured device (GPU in
the paper). verifier = training M0; scientists = sum over the 20 sequential runs; judge =
training J + scoring 20 submissions; total = sum of the three.

Memory: peak RSS via OS-level accounting, with M0, each scientist and the judge each executed in
its own freshly spawned CPU-only subprocess. scientists = sum of the 20 subprocess peaks; total =
sum of the three stage figures (not a simultaneous peak).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from ..config import Config
from ..data import load_task_data
from ..judge import train_judge, validate
from ..models import ARCHS, width_for_target
from ..training import train_run
from ..utils import Timer, derive_seed, resolve_device
from .common import check_or_write_config

RESULT_PREFIX = "SCICHECK_RESULT "


def _scientist_jobs(cfg: Config, arch: str, size: int, seed: int):
    p = cfg.performance
    for i in range(p.n_honest):
        yield "honest", 0.0, i, derive_seed("perf-honest", arch, size, seed, i)
    for i in range(p.n_malicious):
        yield "malicious", p.severity, i, derive_seed("perf-malicious", arch, size, seed, p.severity, i)


def _done_keys(csv: Path) -> set[tuple]:
    if not csv.exists():
        return set()
    df = pd.read_csv(csv)
    return set(zip(df.arch, df.target_size, df.seed))


def _append(csv: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)


# ----------------------------------------------------------------------------- runtime

def runtime_cell(cfg: Config, arch: str, size: int, seed: int, device: torch.device) -> dict:
    data = load_task_data(ARCHS[arch].dataset, cfg.data_dir, cfg.train_frac, cfg.data_seed)
    width = width_for_target(arch, size)
    ver = train_run(arch, width, *data.verifier, cfg.training,
                    derive_seed("perf-verifier", arch, size, seed), device)
    sci_seconds, submissions = 0.0, []
    for _role, flip, _i, run_seed in _scientist_jobs(cfg, arch, size, seed):
        r = train_run(arch, width, *data.scientist, cfg.training, run_seed, device, flip)
        sci_seconds += r.seconds
        submissions.append(r.snapshots)
    with Timer(device) as tj:
        judge = train_judge(ver.snapshots, cfg.judge, cfg.validate.rho, device,
                            derive_seed("perf-judge", arch, size, seed))
        for s in submissions:
            validate(judge, s, cfg.validate.theta)
    return dict(arch=arch, target_size=size, n_params=ver.n_params, seed=seed, device=str(device),
                verifier_s=ver.seconds, scientists_s=sci_seconds, judge_s=tj.seconds,
                total_s=ver.seconds + sci_seconds + tj.seconds)


def run_runtime(cfg: Config, device: torch.device) -> None:
    out = cfg.out_path / "performance"
    out.mkdir(parents=True, exist_ok=True)
    for seed in cfg.seeds:
        for arch in cfg.architectures:
            csv = out / f"runtime_{arch}.csv"  # one file per architecture: parallel jobs never share a file
            done = _done_keys(csv)
            for size in cfg.sizes:
                if (arch, size, seed) in done:
                    continue
                row = runtime_cell(cfg, arch, size, seed, device)
                _append(csv, row)
                print(f"[runtime {arch} {size} s{seed}] total {row['total_s']:.1f}s "
                      f"(M0 {row['verifier_s']:.1f} + sci {row['scientists_s']:.1f} + J {row['judge_s']:.1f})",
                      flush=True)


# ----------------------------------------------------------------------------- memory

def _subprocess(job: dict, workdir: Path) -> dict:
    """Run one stage in a fresh CPU-only interpreter and return its self-reported peak RSS."""
    job_file = workdir / f"job_{job['stage']}_{job.get('tag', '')}.json"
    job_file.write_text(json.dumps(job))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "-m", "scicheck.worker", str(job_file)],
                          capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[2])
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed ({job['stage']}):\n{proc.stderr[-4000:]}")
    line = next(l for l in reversed(proc.stdout.splitlines()) if l.startswith(RESULT_PREFIX))
    return json.loads(line[len(RESULT_PREFIX):])


def memory_cell(cfg: Config, arch: str, size: int, seed: int) -> dict:
    width = width_for_target(arch, size)
    common = dict(arch=arch, width=width, data_dir=str(Path(cfg.data_dir).resolve()),
                  train_frac=cfg.train_frac, data_seed=cfg.data_seed, training=asdict(cfg.training))
    with tempfile.TemporaryDirectory(prefix="scicheck_mem_") as tmp:
        tmp = Path(tmp)
        ver_npy = tmp / "verifier.npy"
        ver = _subprocess(dict(common, stage="train", tag="verifier", partition="verifier", flip=0.0,
                               seed=derive_seed("perf-verifier", arch, size, seed), out=str(ver_npy)), tmp)
        sci_mb, subs = 0.0, []
        for role, flip, i, run_seed in _scientist_jobs(cfg, arch, size, seed):
            npy = tmp / f"{role}_{i}.npy"
            r = _subprocess(dict(common, stage="train", tag=f"{role}{i}", partition="scientist",
                                 flip=flip, seed=run_seed, out=str(npy)), tmp)
            sci_mb += r["peak_rss_mb"]
            subs.append(str(npy))
        j = _subprocess(dict(stage="judge", tag="judge", verifier=str(ver_npy), submissions=subs,
                             judge=asdict(cfg.judge), rho=cfg.validate.rho, theta=cfg.validate.theta,
                             seed=derive_seed("perf-judge", arch, size, seed)), tmp)
    return dict(arch=arch, target_size=size, n_params=ver["n_params"], seed=seed,
                verifier_mb=ver["peak_rss_mb"], scientists_mb=sci_mb, judge_mb=j["peak_rss_mb"],
                total_mb=ver["peak_rss_mb"] + sci_mb + j["peak_rss_mb"])


def run_memory(cfg: Config) -> None:
    out = cfg.out_path / "performance"
    out.mkdir(parents=True, exist_ok=True)
    for seed in cfg.seeds:
        for arch in cfg.architectures:
            csv = out / f"memory_{arch}.csv"
            done = _done_keys(csv)
            for size in cfg.sizes:
                if (arch, size, seed) in done:
                    continue
                row = memory_cell(cfg, arch, size, seed)
                _append(csv, row)
                print(f"[memory {arch} {size} s{seed}] total {row['total_mb']:.0f} MB "
                      f"(M0 {row['verifier_mb']:.0f} + sci {row['scientists_mb']:.0f} + J {row['judge_mb']:.0f})",
                      flush=True)


def run_performance(cfg: Config, sweeps: list[str] | None = None, device: str | None = None) -> None:
    check_or_write_config(cfg, cfg.out_path, "config.performance.json")
    sweeps = sweeps or list(cfg.performance.sweeps)
    if "runtime" in sweeps:
        dev = resolve_device(device or cfg.device)
        if dev.type != "cuda":
            print("note: runtime sweep is running on CPU; the paper's runtimes were measured on GPU", flush=True)
        run_runtime(cfg, dev)
    if "memory" in sweeps:
        run_memory(cfg)
