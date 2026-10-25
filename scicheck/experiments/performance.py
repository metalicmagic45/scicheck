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
from ..judge import spot_errors, spot_tolerance, train_judge, validate
from ..models import ARCHS, width_for_target
from ..storage import checkpoint_storage
from ..training import train_run, verifier_steps
from ..utils import Timer, derive_seed, resolve_device
from .common import check_or_write_config, spot_plan

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
    ver_seed = derive_seed("perf-verifier", arch, size, seed)
    ver = train_run(arch, width, *data.verifier, cfg.training, ver_seed, device,
                    steps=verifier_steps(cfg.verifier_steps, cfg.training, data.n_train), spot=spot_plan(cfg, ver_seed))
    sci_seconds, honest_seconds, submissions, honest_segment = 0.0, [], [], None
    for role, flip, _i, run_seed in _scientist_jobs(cfg, arch, size, seed):
        r = train_run(arch, width, *data.scientist, cfg.training, run_seed, device,
                      "signflip" if flip > 0 else None, flip, spot=spot_plan(cfg, run_seed))
        sci_seconds += r.seconds
        if role == "honest":
            honest_seconds.append(r.seconds)
            if honest_segment is None and r.segments:
                honest_segment = r.segments[0]
        submissions.append(r)
    storage = {}
    if honest_segment is not None:
        bs = cfg.training.batch_size
        storage = checkpoint_storage(honest_segment, cfg.training.epochs * -(-data.n_train // bs), data.n_train,
                                     bs, cfg.validate.spot_steps, cfg.validate.spot_segments)
    with Timer(device) as tj:
        judge = train_judge(ver.snapshots, cfg.judge, cfg.validate.rho, device,
                            derive_seed("perf-judge", arch, size, seed))
    vcfg = cfg.validate
    spe = -(-data.n_train // cfg.training.batch_size) if vcfg.replay else None
    with Timer(device) as ts:
        for r in submissions:
            validate(judge, r.snapshots, vcfg.theta, spe, vcfg.replay_tol, vcfg.psi_max)
    score_s = ts.seconds / len(submissions) if submissions else float("nan")
    calib_s = spot_s = 0.0
    if vcfg.spot_segments:
        with Timer(device) as tc:
            spot_tolerance(spot_errors(arch, width, *data.verifier, ver.segments, cfg.training, device),
                           vcfg.spot_margin, vcfg.spot_floor)
        with Timer(device) as tsp:
            for r in submissions:
                spot_errors(arch, width, *data.scientist, r.segments, cfg.training, device)
        calib_s, spot_s = tc.seconds, tsp.seconds / len(submissions)
    return dict(arch=arch, target_size=size, n_params=ver.n_params, seed=seed, device=str(device),
                verifier_s=ver.seconds, scientists_s=sci_seconds, judge_s=tj.seconds + ts.seconds,
                judge_train_s=tj.seconds, score_s=score_s, spot_calib_s=calib_s, spot_s=spot_s,
                retrain_s=sum(honest_seconds) / len(honest_seconds) if honest_seconds else float("nan"),
                verify_s=ver.seconds + tj.seconds + score_s + calib_s + spot_s,
                total_s=ver.seconds + sci_seconds + tj.seconds + ts.seconds + calib_s + spot_s * len(submissions),
                **storage)


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
                      f"+ score {row['score_s']:.1f} + spot-check {row['spot_calib_s'] + row['spot_s']:.1f})"
                      + (f" | checkpoints {row['keep_all_mb']:.0f} MB kept all vs {row['bounded_peak_mb']:.0f} MB "
                         f"one epoch at a time" if "keep_all_mb" in row else ""),
                      flush=True)


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
        ver_npy, ver_seed = tmp / "verifier.npy", derive_seed("perf-verifier", arch, size, seed)
        ver = _subprocess(dict(common, stage="train", tag="verifier", partition="verifier", flip=0.0,
                               seed=ver_seed, out=str(ver_npy), spot=spot_plan(cfg, ver_seed),
                               segments=str(tmp / "verifier.seg")), tmp)
        sci_peaks, subs, segs = [], [], []
        for role, flip, i, run_seed in _scientist_jobs(cfg, arch, size, seed):
            npy, seg = tmp / f"{role}_{i}.npy", tmp / f"{role}_{i}.seg"
            r = _subprocess(dict(common, stage="train", tag=f"{role}{i}", partition="scientist",
                                 flip=flip, seed=run_seed, out=str(npy), spot=spot_plan(cfg, run_seed),
                                 segments=str(seg)), tmp)
            sci_peaks.append(r["peak_rss_mb"])
            subs.append(str(npy))
            segs.append(str(seg))
        j = _subprocess(dict(stage="judge", tag="judge", verifier=str(ver_npy), submissions=subs,
                             judge=asdict(cfg.judge), rho=cfg.validate.rho, theta=cfg.validate.theta,
                             seed=derive_seed("perf-judge", arch, size, seed)), tmp)
        spot_mb = float("nan")
        if cfg.validate.spot_segments:
            spot_mb = _subprocess(dict(common, stage="spot", tag="spot", verifier=str(tmp / "verifier.seg"),
                                       submissions=segs, margin=cfg.validate.spot_margin,
                                       floor=cfg.validate.spot_floor), tmp)["peak_rss_mb"]
    sci_mb = sum(sci_peaks)
    stage_peaks = [ver["peak_rss_mb"], j["peak_rss_mb"], *sci_peaks] + ([spot_mb] if spot_mb == spot_mb else [])
    return dict(arch=arch, target_size=size, n_params=ver["n_params"], seed=seed,
                verifier_mb=ver["peak_rss_mb"], scientists_mb=sci_mb, judge_mb=j["peak_rss_mb"], spot_mb=spot_mb,
                scientist_mb=sci_mb / len(sci_peaks) if sci_peaks else float("nan"),
                peak_mb=max(stage_peaks),
                total_mb=ver["peak_rss_mb"] + sci_mb + j["peak_rss_mb"] + (spot_mb if spot_mb == spot_mb else 0.0))


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
                      f"(M0 {row['verifier_mb']:.0f}, scientist {row['scientist_mb']:.0f}, J {row['judge_mb']:.0f}, "
                      f"spot-check {row['spot_mb']:.0f})", flush=True)


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
