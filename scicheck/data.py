from __future__ import annotations

import io
import shutil
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PETIMAGES_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
PETIMAGES_ZIP = "kagglecatsanddogs_5340.zip"
PLACEMENT_KAGGLE = "sahilislam007/college-student-placement-factors-dataset"
PLACEMENT_CSV = "placement.csv"

N_PETIMAGES = 1500
N_MNIST = 3000
IMG_SIZE = 32


@dataclass
class TaskData:
    X: np.ndarray
    y: np.ndarray
    n_verifier: int

    @property
    def verifier(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X[: self.n_verifier], self.y[: self.n_verifier]

    @property
    def scientist(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y


def load_task_data(dataset: str, data_dir: str | Path, train_frac: float, data_seed: int = 0) -> TaskData:
    data_dir = Path(data_dir)
    if dataset == "placement":
        X, y = load_placement(data_dir)
    elif dataset == "petimages":
        X, y = load_petimages(data_dir, seed=data_seed)
    elif dataset == "mnist":
        X = load_mnist(data_dir, seed=data_seed)
        y = X
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return TaskData(X=X, y=y, n_verifier=int(round(train_frac * len(X))))


def to_tensors(X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.as_tensor(X, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.float32, device=device))


def _find_placement_csv(data_dir: Path) -> Path:
    p = data_dir / "raw" / PLACEMENT_CSV
    if p.exists():
        return p
    hits = sorted((data_dir / "raw").glob("*lacement*.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"Placement CSV not found at {p}. Run `python -m scicheck download`, or download "
        f"https://www.kaggle.com/datasets/{PLACEMENT_KAGGLE} and save the CSV as {p}."
    )


def load_placement(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(_find_placement_csv(data_dir))
    label = next(c for c in df.columns if c.strip().lower() == "placement")
    df = df.drop(columns=[c for c in df.columns if c.strip().lower() in ("college_id", "student_id", "id")])

    def to_numeric(col):
        if not pd.api.types.is_numeric_dtype(col):
            mapped = col.astype(str).str.strip().str.lower().map({"yes": 1.0, "no": 0.0})
            if mapped.isna().any():
                raise ValueError(f"column {col.name!r} has non Yes/No strings")
            return mapped
        return col.astype(float)

    df = df.apply(to_numeric)
    y = df.pop(label).to_numpy(np.float32)
    X = df.to_numpy(np.float32)
    if X.shape[1] != 8:
        raise ValueError(f"expected 8 predictors after preprocessing, got {X.shape[1]}: {list(df.columns)}")
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return X.astype(np.float32), y


def load_petimages(data_dir: Path, n: int = N_PETIMAGES, size: int = IMG_SIZE, seed: int = 0):
    cache = data_dir / "processed" / f"petimages_{n}_{size}px_seed{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"].astype(np.float32) / 255.0, z["y"].astype(np.float32)

    from PIL import Image

    rng = np.random.default_rng(seed)
    per_class = n // 2
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

    images, labels = [], []
    for label, cls in enumerate(("Cat", "Dog")):
        files = listing[cls]
        got = 0
        for i in rng.permutation(len(files)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img = Image.open(io.BytesIO(read(files[i])))
                    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
            except Exception:
                continue
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(label)
            got += 1
            if got == per_class:
                break
    X = np.stack(images).transpose(0, 3, 1, 2)
    y = np.asarray(labels, dtype=np.uint8)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X.astype(np.float32) / 255.0, y.astype(np.float32)


def load_mnist(data_dir: Path, n: int = N_MNIST, seed: int = 0) -> np.ndarray:
    cache = data_dir / "processed" / f"mnist_{n}_seed{seed}.npz"
    if cache.exists():
        return np.load(cache)["X"].astype(np.float32) / 255.0
    from torchvision.datasets import MNIST

    ds = MNIST(root=str(data_dir / "raw"), train=True, download=True)
    idx = np.random.default_rng(seed).choice(len(ds.data), size=n, replace=False)
    X = ds.data.numpy()[idx].reshape(n, -1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X)
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


def download_all(data_dir: str | Path, data_seed: int = 0) -> None:
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print("[mnist]")
    load_mnist(data_dir, seed=data_seed)

    print("[petimages]")
    if not (raw / PETIMAGES_ZIP).exists() and not (raw / "PetImages").exists():
        _download(PETIMAGES_URL, raw / PETIMAGES_ZIP)
    X, y = load_petimages(data_dir, seed=data_seed)
    print(f"  {len(X)} images, {int(y.sum())} dogs")

    print("[placement]")
    try:
        _find_placement_csv(data_dir)
    except FileNotFoundError:
        try:
            import kagglehub

            path = Path(kagglehub.dataset_download(PLACEMENT_KAGGLE))
            csv = next(path.rglob("*.csv"))
            shutil.copy(csv, raw / PLACEMENT_CSV)
        except Exception as e:  # noqa: BLE001
            print(f"  automatic download failed ({type(e).__name__}: {e})")
            print(f"  -> download https://www.kaggle.com/datasets/{PLACEMENT_KAGGLE}")
            print(f"     and save the CSV as {raw / PLACEMENT_CSV}")
            return
    X, y = load_placement(data_dir)
    print(f"  {X.shape[0]} rows x {X.shape[1]} predictors, {y.mean():.1%} placed")
