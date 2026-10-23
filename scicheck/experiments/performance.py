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
from ..training import train_run, verifier_steps
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


def runtime_cell(cfg: Config, arch: str, size: int, seed: int, device: torch.device) -> dict:
    data = load_task_data(ARCHS[arch].dataset, cfg.data_dir, cfg.train_frac, cfg.data_seed, cfg.data)
    width = width_for_target(arch, size)
    ver = train_run(arch, width, *data.verifier, cfg.training, derive_seed("perf-verifier", arch, size, seed), device,
                    steps=verifier_steps(cfg.verifier_steps, cfg.training, data.n_train))
    sci_seconds, honest_seconds, submissions = 0.0, [], []
    for role, flip, _i, run_seed in _scientist_jobs(cfg, arch, size, seed):
        r = train_run(arch, width, *data.scientist, cfg.training, run_seed, device,
                      "signflip" if flip > 0 else None, flip)
        sci_seconds += r.seconds
        if role == "honest":
            honest_seconds.append(r.seconds)
        submissions.append(r.snapshots)
    with Timer(device) as tj:
        judge = train_judge(ver.snapshots, cfg.judge, cfg.validate.rho, device,
                            derive_seed("perf-judge", arch, size, seed))
    vcfg = cfg.validate
    spe = -(-data.n_train // cfg.training.batch_size) if vcfg.replay else None
    with Timer(device) as ts:
        for s in submissions:
            validate(judge, s, vcfg.theta, spe, vcfg.replay_tol, vcfg.psi_max)
    score_s = ts.seconds / len(submissions) if submissions else float("nan")
    # Retraining: what checking a run by training the blueprint again would cost, i.e. one honest full-data run.
    # SciCheck verification: the verifier's reference run, fitting the judge, and scoring one submission.
    return dict(arch=arch, target_size=size, n_params=ver.n_params, seed=seed, device=str(device),
                verifier_s=ver.seconds, scientists_s=sci_seconds, judge_s=tj.seconds + ts.seconds,
                judge_train_s=tj.seconds, score_s=score_s,
                retrain_s=sum(honest_seconds) / len(honest_seconds) if honest_seconds else float("nan"),
                verify_s=ver.seconds + tj.seconds + score_s,
                total_s=ver.seconds + sci_seconds + tj.seconds + ts.seconds)


def run_runtime(cfg: Config, device: torch.device) -> None:
    out = cfg.out_path / "performance"
    out.mkdir(parents=True, exist_ok=True)
    for seed in cfg.seeds:
        for arch in cfg.architectures:
            csv = out / f"runtime_{arch}.csv"
            done = _done_keys(csv)
            for size in cfg.sizes:
                if (arch, size, seed) in done:
                    continue
                row = runtime_cell(cfg, arch, size, seed, device)
                _append(csv, row)
                print(f"[runtime {arch} {size} s{seed}] retraining {row['retrain_s']:.1f}s vs SciCheck "
                      f"{row['verify_s']:.1f}s (M0 {row['verifier_s']:.1f} + J {row['judge_train_s']:.1f} "
                      f"+ score {row['score_s']:.1f})", flush=True)


def _subprocess(job: dict, workdir: Path) -> dict:
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
                  train_frac=cfg.train_frac, data_seed=cfg.data_seed, data=asdict(cfg.data),
                  training=asdict(cfg.training), verifier_steps=cfg.verifier_steps)
    with tempfile.TemporaryDirectory(prefix="scicheck_mem_") as tmp:
        tmp = Path(tmp)
        ver_npy = tmp / "verifier.npy"
        ver = _subprocess(dict(common, stage="train", tag="verifier", partition="verifier", flip=0.0,
                               seed=derive_seed("perf-verifier", arch, size, seed), out=str(ver_npy)), tmp)
        sci_peaks, subs = [], []
        for role, flip, i, run_seed in _scientist_jobs(cfg, arch, size, seed):
            npy = tmp / f"{role}_{i}.npy"
            r = _subprocess(dict(common, stage="train", tag=f"{role}{i}", partition="scientist",
                                 flip=flip, seed=run_seed, out=str(npy)), tmp)
            sci_peaks.append(r["peak_rss_mb"])
            subs.append(str(npy))
        j = _subprocess(dict(stage="judge", tag="judge", verifier=str(ver_npy), submissions=subs,
                             judge=asdict(cfg.judge), rho=cfg.validate.rho, theta=cfg.validate.theta,
                             seed=derive_seed("perf-judge", arch, size, seed)), tmp)
    sci_mb = sum(sci_peaks)
    # peak_mb is the most any one stage needs: the stages run one after another, so it is what a machine must have.
    return dict(arch=arch, target_size=size, n_params=ver["n_params"], seed=seed,
                verifier_mb=ver["peak_rss_mb"], scientists_mb=sci_mb, judge_mb=j["peak_rss_mb"],
                scientist_mb=sci_mb / len(sci_peaks) if sci_peaks else float("nan"),
                peak_mb=max([ver["peak_rss_mb"], j["peak_rss_mb"], *sci_peaks]),
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
                print(f"[memory {arch} {size} s{seed}] peak {row['peak_mb']:.0f} MB "
                      f"(M0 {row['verifier_mb']:.0f}, scientist {row['scientist_mb']:.0f}, J {row['judge_mb']:.0f})",
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
