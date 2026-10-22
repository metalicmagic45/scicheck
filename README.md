# SciCheck

Code for *SciCheck: A Framework for Verifying Deep Learning Viability during Training*.
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) explains the method.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CPU: .../whl/cpu
pip install -r requirements.txt
python -m scicheck download
```

`download` puts MNIST, PetImages and the placement CSV in `data/`. If Kaggle asks for credentials,
download the CSV from https://www.kaggle.com/datasets/sahilislam007/college-student-placement-factors-dataset
and save it as `data/raw/placement.csv`.

## Run

Quick check (a few minutes):

```bash
python -m pytest tests -q
python -m scicheck accuracy --config configs/smoke.yaml
python -m scicheck report   --config configs/smoke.yaml
```

Full experiments:

```bash
python -m scicheck accuracy    --config configs/paper.yaml --workers 8 --devices cuda:0
python -m scicheck performance --config configs/paper.yaml
python -m scicheck report      --config configs/paper.yaml
```

`configs/paper.yaml` currently runs 100k-500k parameter models with 1 seed and 5 honest + 5 malicious
runs. The paper used 4k-12k, 5 seeds and 10 + 10. `jobs/` has SLURM and plain shell versions of
these steps. If `accuracy` is stopped, rerun the same command and it continues where it left off.

## Output

`accuracy` prints one line per finished (architecture, size, seed), for example:

```
(3/15) [mlp 100000 s0 cuda:0] done in 812s | honest rejected 1/5 | malicious rejected 25/25
```

Everything is written to the config's `out_dir` (e.g. `results/paper/`):

| path | contents |
|---|---|
| `runs.csv` | one row per scientist run: role, flip, phi, eta_s, both check results, accept |
| `units/<arch>_<size>_s<seed>/` | that unit's runs, judge (`judge.pt`), snapshots, training diagnostics |
| `performance/` | runtime and peak memory per (arch, size) |
| `report/report.md` | Tables IV-VI, Figure 2, ablation, θ/ρ sensitivity, plus CSV/PNG of each |

What to expect (from the paper's 4k-12k runs): MLP and CNN reject nearly all malicious runs at flip ≥ 0.3, and miss many at
flip 0.1. The autoencoder rejects most honest runs as well. The smoke config is too small for
meaningful numbers; it only checks that everything runs end to end.
