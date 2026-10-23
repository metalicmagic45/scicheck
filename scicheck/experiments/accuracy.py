from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..attacks import ATTACKS
from ..baselines import fit_baselines
from ..config import Config
from ..data import load_task_data
from ..judge import train_judge, validate
from ..models import ARCHS, width_for_target
from ..training import train_run, verifier_steps
from ..utils import Timer, derive_seed, resolve_device, write_json
from .common import check_or_write_config, run_jobs


def _last(r, key: str) -> float:
    return r.history[-1].get(key, float("nan")) if r.history else float("nan")


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
    data = load_task_data(spec.dataset, cfg.data_dir, cfg.train_frac, cfg.data_seed, cfg.data)
    width = width_for_target(arch, size)

    ver = train_run(arch, width, *data.verifier, cfg.training, derive_seed("verifier", arch, size, seed), device,
                    eval_data=data.test, steps=verifier_steps(cfg.verifier_steps, cfg.training, data.n_train))
    with Timer(device) as tj:
        judge = train_judge(ver.snapshots, cfg.judge, cfg.validate.rho, device,
                            derive_seed("judge", arch, size, seed))
    print(f"{tag} M0 ({ver.n_params} params, {len(ver.snapshots)} snapshots) {ver.seconds:.0f}s | "
          f"J {tj.seconds:.0f}s, {judge.diagnostics['epochs_run']} ep, tau={judge.tau:.4g}, "
          f"eta_v={judge.eta_v:.4g} | loss train {ver.final_loss:.4f} / held-out {_last(ver, 'eval_loss'):.4f}",
          flush=True)
    with Timer() as tb:
        baselines = fit_baselines(ver.snapshots, judge, cfg.judge, derive_seed("baselines", arch, size, seed))

    base = dict(arch=arch, target_size=size, n_params=ver.n_params, width=width, seed=seed,
                tau=judge.tau, eta_v=judge.eta_v, gamma=judge.gamma)
    rows, snaps = [], {"verifier": ver.snapshots}
    history = [dict(role="verifier", attack="none", severity=0.0, run_idx=0, **h) for h in ver.history]
    vcfg = cfg.validate
    spe = -(-data.n_train // cfg.training.batch_size) if vcfg.replay else None

    def scientist(attack: str | None, severity: float, idx: int, run_seed: int) -> None:
        r = train_run(arch, width, *data.scientist, cfg.training, run_seed, device, attack, severity,
                      eval_data=data.test)
        v = validate(judge, r.snapshots, vcfg.theta, spe, vcfg.replay_tol, vcfg.psi_max)
        role, attack = ("malicious", attack) if attack else ("honest", "none")
        rows.append(dict(base, role=role, attack=attack, severity=severity, run_idx=idx, run_seed=run_seed,
                         n_snapshots=len(r.snapshots), final_loss=r.final_loss,
                         eval_loss=_last(r, "eval_loss"), eval_acc=_last(r, "eval_acc"), train_seconds=r.seconds,
                         phi=v.phi, eta_s=v.eta_s, at_pass=v.at_pass, eta_pass=v.eta_pass, accept=v.accept,
                         **({"psi": v.psi, "replay_pass": v.replay_pass} if spe else {}),
                         **{f"phi_{k}": b.phi(judge, r.snapshots) for k, b in baselines.items()}))
        history.extend(dict(role=role, attack=attack, severity=severity, run_idx=idx, **h) for h in r.history)
        if cfg.save_snapshots:
            snaps[f"honest_{idx}" if role == "honest" else f"{attack}_{severity:g}_{idx}"] = r.snapshots

    for i in range(cfg.n_honest):
        scientist(None, 0.0, i, derive_seed("honest", arch, size, seed, i))
    for attack in cfg.attacks:
        for sev in cfg.severities:
            for i in range(cfg.n_malicious):
                scientist(attack, float(sev), i, derive_seed("malicious", attack, arch, size, seed, sev, i))

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.save_snapshots:
        np.savez_compressed(out / "snapshots.npz", **snaps)
    judge.save(out / "judge.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    write_json(out / "meta.json", dict(base, device=str(device), verifier_seconds=ver.seconds,
                                       verifier_final_loss=ver.final_loss,
                                       verifier_eval_loss=_last(ver, "eval_loss"),
                                       data_rows_scientist=data.n_train, data_rows_verifier=data.n_verifier,
                                       data_rows_test=len(data.X) - data.n_train, judge_seconds=tj.seconds,
                                       baseline_seconds=tb.seconds,
                                       **{f"tau_{k}": b.tau for k, b in baselines.items()},
                                       unit_seconds=time.perf_counter() - t0, **judge.diagnostics))
    tmp = out / "runs.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(out / "runs.csv")

    h, m = df[df.role == "honest"], df[df.role == "malicious"]
    per_attack = ", ".join(f"{a} {(~g.accept).sum()}/{len(g)}" for a, g in m.groupby("attack", sort=False))
    return (f"{tag} done in {time.perf_counter() - t0:.0f}s | honest rejected {(~h.accept).sum()}/{len(h)} | "
            f"malicious rejected {per_attack}")


def run_accuracy(cfg: Config, workers: int = 1, devices: list[str] | None = None,
                 threads: int | None = None) -> pd.DataFrame:
    unknown = sorted(set(cfg.attacks) - set(ATTACKS))
    if unknown:
        raise SystemExit(f"unknown attack(s) {unknown} in the config; expected some of {list(ATTACKS)}")
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
    tmp = Path(out) / f"runs.csv.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, Path(out) / "runs.csv")
    return df
