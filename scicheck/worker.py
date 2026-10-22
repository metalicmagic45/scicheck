from __future__ import annotations

import json
import sys

import numpy as np
import torch

from .config import JudgeConfig, TrainingConfig
from .data import load_task_data
from .judge import train_judge, validate
from .models import ARCHS
from .training import train_run
from .utils import peak_rss_mb

RESULT_PREFIX = "SCICHECK_RESULT "


def main(job_path: str) -> None:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    device = torch.device("cpu")
    result: dict = {}

    if job["stage"] == "train":
        data = load_task_data(ARCHS[job["arch"]].dataset, job["data_dir"], job["train_frac"], job["data_seed"])
        X, y = data.verifier if job["partition"] == "verifier" else data.scientist
        r = train_run(job["arch"], job["width"], X, y, TrainingConfig(**job["training"]),
                      job["seed"], device, job["flip"])
        np.save(job["out"], r.snapshots)
        result = dict(seconds=r.seconds, n_params=r.n_params, n_snapshots=len(r.snapshots))
    elif job["stage"] == "judge":
        cfg = JudgeConfig(**{**job["judge"], "split": tuple(job["judge"]["split"])})
        judge = train_judge(np.load(job["verifier"]), cfg, job["rho"], device, job["seed"])
        for path in job["submissions"]:
            validate(judge, np.load(path), job["theta"])
    else:
        raise ValueError(f"unknown stage {job['stage']!r}")

    result["peak_rss_mb"] = peak_rss_mb()
    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
