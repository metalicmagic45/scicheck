"""§VI-A accuracy sweep: architectures x sizes x severities x seeds, 10 honest + 10 malicious
scientist runs per cell.

The unit of work is one (architecture, size, seed): one verifying-scientist model M0, one judge
model J, the honest runs, and the malicious runs for every severity. Honest runs never depend on
the severity, so they are trained once per unit and counted in every severity cell (the paper's
MLP/AE FalsePositive is identical across severities for the same reason). Each unit writes
``units/<arch>_<size>_s<seed>/runs.csv`` last, so an interrupted sweep resumes where it stopped.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data import load_task_data
from ..judge import train_judge, validate
from ..models import ARCHS, width_for_target
from ..training import train_run
from ..utils import Timer, derive_seed, resolve_device, write_json
from .common import check_or_write_config, run_jobs


def unit_dir(cfg: Config, arch: str, size: int, seed: int) -> Path:
    return cfg.out_path / "units" / f"{arch}_{size}_s{seed}"


def run_unit(cfg: Config, arch: str, size: int, seed: int, device: torch.device) -> str:
    out = unit_dir(cfg, arch, size, seed)
    tag = f"[{arch} {size} s{seed} {device}]"
    if (out / "runs.csv").exists():
        return f"{tag} already done"
    if cfg.deterministic:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    t0 = time.perf_counter()
    spec = ARCHS[arch]
    data = load_task_data(spec.dataset, cfg.data_dir, cfg.train_frac, cfg.data_seed)
    width = width_for_target(arch, size)

    # Verifying scientist phase (Alg. 1): reference model M0 on the first train_frac of the data.
    ver = train_run(arch, width, *data.verifier, cfg.training, derive_seed("verifier", arch, size, seed), device)
    with Timer(device) as tj:
        judge = train_judge(ver.snapshots, cfg.judge, cfg.validate.rho, device,
                            derive_seed("judge", arch, size, seed))
    print(f"{tag} M0 ({ver.n_params} params, {len(ver.snapshots)} snapshots) {ver.seconds:.0f}s | "
          f"J {tj.seconds:.0f}s, {judge.diagnostics['epochs_run']} ep, tau={judge.tau:.4g}, "
          f"eta_v={judge.eta_v:.4g}", flush=True)

    base = dict(arch=arch, target_size=size, n_params=ver.n_params, width=width, seed=seed,
                tau=judge.tau, eta_v=judge.eta_v, gamma=judge.gamma)
    rows, snaps = [], {"verifier": ver.snapshots}

    # Scientist phase + verification phase (Alg. 2): every scientist trains on the full dataset.
    def scientist(role: str, flip: float, idx: int, run_seed: int) -> None:
        r = train_run(arch, width, *data.scientist, cfg.training, run_seed, device, flip)
        v = validate(judge, r.snapshots, cfg.validate.theta)
        rows.append(dict(base, role=role, flip=flip, run_idx=idx, run_seed=run_seed,
                         n_snapshots=len(r.snapshots), final_loss=r.final_loss, train_seconds=r.seconds,
                         phi=v.phi, eta_s=v.eta_s, at_pass=v.at_pass, eta_pass=v.eta_pass, accept=v.accept))
        if cfg.save_snapshots:
            snaps[f"{role}_{flip:g}_{idx}"] = r.snapshots

    for i in range(cfg.n_honest):
        scientist("honest", 0.0, i, derive_seed("honest", arch, size, seed, i))
    for flip in cfg.severities:
        for i in range(cfg.n_malicious):
            scientist("malicious", float(flip), i, derive_seed("malicious", arch, size, seed, flip, i))

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.save_snapshots:
        np.savez_compressed(out / "snapshots.npz", **snaps)
    judge.save(out / "judge.pt")
    write_json(out / "meta.json", dict(base, device=str(device), verifier_seconds=ver.seconds,
                                       verifier_final_loss=ver.final_loss, judge_seconds=tj.seconds,
                                       unit_seconds=time.perf_counter() - t0, **judge.diagnostics))
    tmp = out / "runs.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(out / "runs.csv")  # completion marker

    h, m = df[df.role == "honest"], df[df.role == "malicious"]
    return (f"{tag} done in {time.perf_counter() - t0:.0f}s | honest rejected {(~h.accept).sum()}/{len(h)} | "
            f"malicious rejected {(~m.accept).sum()}/{len(m)}")


def run_accuracy(cfg: Config, workers: int = 1, devices: list[str] | None = None,
                 threads: int | None = None) -> pd.DataFrame:
    check_or_write_config(cfg, cfg.out_path)
    devices = devices or [str(resolve_device(cfg.device))]
    units =[(cfg, a, s, seed) for seed in cfg.seeds for a in cfg.architectures for s in cfg.sizes]
    todo = [u for u in units if not (unit_dir(*u) / "runs.csv").exists()]
    print(f"{len(units)} units ({len(units) - len(todo)} already done), workers={workers}, "
          f"devices={devices}", flush=True)
    for i, msg in enumerate(run_jobs(run_unit, todo, workers, devices, threads), 1):
        print(f"({i}/{len(todo)}) {msg}", flush=True)
    return collect_runs(cfg.out_path)


def collect_runs(out: Path) -> pd.DataFrame:
    files = sorted((Path(out) / "units").glob("*/runs.csv"))
    if not files:
        raise FileNotFoundError(f"no finished units under {out / 'units'}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Several per-architecture jobs may finish at once: write via a private temp file, then rename.
    tmp = Path(out) / f"runs.csv.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, Path(out) / "runs.csv")
    return df
