from __future__ import annotations

import gzip
import io
import os
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig

PETIMAGES_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
PETIMAGES_ZIP = "kagglecatsanddogs_5340.zip"
HIGGS_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
HIGGS_ZIP = "higgs.zip"
HIGGS_FEATURES = 28

IMG_SIZE = 32


@dataclass
class TaskData:
    X: np.ndarray
    y: np.ndarray
    n_train: int
    n_verifier: int

    @property
    def verifier(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X[: self.n_verifier], self.y[: self.n_verifier]

    @property
    def scientist(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X[: self.n_train], self.y[: self.n_train]

    @property
    def test(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X[self.n_train:], self.y[self.n_train:]


def load_task_data(dataset: str, data_dir: str | Path, train_frac: float, data_seed: int = 0,
                   dcfg: DataConfig | None = None) -> TaskData:
    data_dir, dcfg = Path(data_dir), dcfg or DataConfig()
    if dataset == "higgs":
        X, y = load_higgs(data_dir, n=dcfg.higgs_rows, seed=data_seed)
    elif dataset == "petimages":
        X, y = load_petimages(data_dir, n=dcfg.petimages, seed=data_seed)
    elif dataset == "mnist":
        X = load_mnist(data_dir, n=dcfg.mnist, seed=data_seed)
        y = X
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    n_train = len(X) - int(round(dcfg.test_frac * len(X)))
    return TaskData(X=X, y=y, n_train=n_train, n_verifier=int(round(train_frac * n_train)))


def to_tensors(X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.as_tensor(X, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.float32, device=device))


def _save_cache(cache: Path, **arrays) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, cache)


def _open_higgs_csv(raw: Path):
    if (raw / "HIGGS.csv").exists():
        return open(raw / "HIGGS.csv", "rb")
    if (raw / "HIGGS.csv.gz").exists():
        return gzip.open(raw / "HIGGS.csv.gz")
    if (raw / HIGGS_ZIP).exists():
        zf = zipfile.ZipFile(raw / HIGGS_ZIP)
        inner = next(x for x in zf.namelist() if x.lower().startswith("higgs.csv"))
        f = zf.open(inner)
        return gzip.open(f) if inner.lower().endswith(".gz") else f
    raise FileNotFoundError(
        f"HIGGS not found in {raw}. Run `python -m scicheck download`, or download {HIGGS_URL} "
        f"and save it as {raw / HIGGS_ZIP} (HIGGS.csv.gz or HIGGS.csv there also work)."
    )


def load_higgs(data_dir: Path, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("data.higgs_rows must be positive (the full file is 11M rows)")
    cache = data_dir / "processed" / f"higgs_{n}_seed{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]

    import pandas as pd

    with _open_higgs_csv(data_dir / "raw") as f:
        df = pd.read_csv(f, header=None, nrows=n, dtype=np.float32)
    if df.shape[1] != HIGGS_FEATURES + 1 or len(df) < n:
        raise ValueError(f"expected {n} rows x {HIGGS_FEATURES + 1} columns, got {df.shape}")
    order = np.random.default_rng(seed).permutation(n)
    y = df.iloc[:, 0].to_numpy(np.float32)[order]
    X = df.iloc[:, 1:].to_numpy(np.float32)[order]
    X = ((X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)).astype(np.float32)
    _save_cache(cache, X=X, y=y)
    return X, y


def load_petimages(data_dir: Path, n: int = 0, size: int = IMG_SIZE, seed: int = 0):
    cache = data_dir / "processed" / f"petimages_{n or 'all'}_{size}px_seed{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"].astype(np.float32) / 255.0, z["y"].astype(np.float32)

    from PIL import Image

    rng = np.random.default_rng(seed)
    per_class = n // 2 if n else None
    src_zip = data_dir / "raw" / PETIMAGES_ZIP
    src_dir = data_dir / "raw" / "PetImages"
    if src_zip.exists():
        zf = zipfile.ZipFile(src_zip)
        names = zf.namelist()
        listing = {c: sorted(x for x in names if f"PetImages/{c}/" in x and x.lower().endswith(".jpg"))
                   for c in ("Cat", "Dog")}
        read = lambda name: zf.read(name)
    elif src_dir.exists():
        listing = {c: sorted(str(p) for p in (src_dir / c).glob("*.jpg")) for c in ("Cat", "Dog")}
        read = lambda name: Path(name).read_bytes()
    else:
        raise FileNotFoundError(f"PetImages not found ({src_zip} or {src_dir}). Run `python -m scicheck download`.")

    by_class = []
    for cls in ("Cat", "Dog"):
        files, imgs = listing[cls], []
        for i in rng.permutation(len(files)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img = Image.open(io.BytesIO(read(files[i])))
                    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
            except Exception:
                continue
            imgs.append(np.asarray(img, dtype=np.uint8))
            if len(imgs) == per_class:
                break
        by_class.append(imgs)
    k = min(len(c) for c in by_class)
    if per_class and k < per_class:
        raise ValueError(f"only {k} readable images in one class, {per_class} requested")
    X = np.stack(by_class[0][:k] + by_class[1][:k]).transpose(0, 3, 1, 2)
    y = np.repeat(np.arange(2, dtype=np.uint8), k)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    _save_cache(cache, X=X, y=y)
    return X.astype(np.float32) / 255.0, y.astype(np.float32)


def load_mnist(data_dir: Path, n: int = 0, seed: int = 0) -> np.ndarray:
    cache = data_dir / "processed" / f"mnist_{n or 'all'}_seed{seed}.npz"
    if cache.exists():
        return np.load(cache)["X"].astype(np.float32) / 255.0
    from torchvision.datasets import MNIST

    ds = MNIST(root=str(data_dir / "raw"), train=True, download=True)
    idx = np.random.default_rng(seed).choice(len(ds.data), size=n or len(ds.data), replace=False)
    X = ds.data.numpy()[idx].reshape(len(idx), -1)
    _save_cache(cache, X=X)
    return X.astype(np.float32) / 255.0


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def download_all(data_dir: str | Path, data_seed: int = 0, dcfg: DataConfig | None = None) -> None:
    data_dir, dcfg = Path(data_dir), dcfg or DataConfig()
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print("[mnist]")
    X = load_mnist(data_dir, n=dcfg.mnist, seed=data_seed)
    print(f"  {len(X)} digits")

    print("[petimages]")
    if not (raw / PETIMAGES_ZIP).exists() and not (raw / "PetImages").exists():
        _download(PETIMAGES_URL, raw / PETIMAGES_ZIP)
    X, y = load_petimages(data_dir, n=dcfg.petimages, seed=data_seed)
    print(f"  {len(X)} images, {int(y.sum())} dogs")

    print("[higgs]")
    if not any((raw / f).exists() for f in (HIGGS_ZIP, "HIGGS.csv.gz", "HIGGS.csv")):
        _download(HIGGS_URL, raw / HIGGS_ZIP)
    X, y = load_higgs(data_dir, n=dcfg.higgs_rows, seed=data_seed)
    print(f"  {X.shape[0]} rows x {X.shape[1]} features, {y.mean():.1%} signal")
