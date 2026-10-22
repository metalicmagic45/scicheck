# SciCheck — reproduction pipeline

Reimplementation of the full experimental pipeline of *SciCheck: A Framework for Verifying Deep
Learning Viability during Training* (BigData'26 submission BigD347), rebuilt from the paper text.

```
scicheck/
  models.py               MLP / CNN / AE blueprints, width -> nearest 4k..12k params   (§VI-A)
  snapshots.py            13 + L per-batch statistics                                    (§V-A, Table II)
  attacks.py              per-batch Bernoulli(flip) gradient sign-flip                   (§VI-A)
  training.py             one training run under a blueprint, recording snapshots       (§IV a/b)
  anomaly_transformer.py  Anomaly Transformer (port of Xu et al.'s reference code)       (§V-B)
  judge.py                Algorithm 1 (train_judge) and Algorithm 2 (validate)           (§V-C, §V-D)
  data.py                 Placement CSV, PetImages 32x32, MNIST subsample                (Table III)
  experiments/accuracy.py     3 archs x 5 sizes x 5 severities x 5 seeds x (10+10)      (§VI-A)
  experiments/performance.py  runtime (in-process, GPU) and memory (isolated subprocesses) (§VI-B)
  worker.py               single-stage subprocess used by the memory sweep
  report.py               Tables IV, V, VI, Figure 2 + ablation and (θ, ρ) sensitivity
configs/paper.yaml        every setting from §VI (choices the paper leaves open are marked NOT STATED)
configs/smoke.yaml        minutes-long end-to-end check on real data
tests/                    fast checks of the implementation against facts stated in the paper
```

## Setup

### GPU server (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
# pick the wheel that matches the driver (`nvidia-smi` shows the max CUDA version supported):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m scicheck info          # should report "cuda available: True" and the param table below
python -m pytest tests -q
```

### Local Windows (development / smoke tests, CPU)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

## Data

```bash
python -m scicheck download
```

* **MNIST** — torchvision download; 3,000 training digits sampled once (`data_seed`).
* **PetImages** — Microsoft's Cats-vs-Dogs zip (~800 MB); 750 cats + 750 dogs decoded straight from
  the zip (corrupt files skipped), resized to 32×32 RGB, shuffled once, cached in `data/processed/`.
* **Placement** — Kaggle *College Student Placement Factors* (10,000 rows). Fetched with
  `kagglehub`; if Kaggle wants credentials, put `kaggle.json` in `~/.kaggle/` or download the CSV
  by hand to `data/raw/placement.csv`. Preprocessing: drop `College_ID`, Yes/No → 1/0, z-score the
  8 predictors, keep file order (the verifier gets the first 3,000 rows).

If the server has no internet access, run `download` locally and copy `data/processed/` and
`data/raw/placement.csv` over. The processed caches are all that the experiments read.

## Running

```bash
python -m scicheck accuracy --config configs/smoke.yaml        # sanity check, a few minutes
python -m scicheck report   --config configs/smoke.yaml

# reduced §VI-A grid (configs/paper.yaml): 15 (arch, size) units, one judge each, 5 honest + 5 malicious
# runs per severity. The paper used 5 seeds and 10+10 runs (75 units, 7,500 scored runs); to restore that
# set seeds: [0,1,2,3,4] and n_honest = n_malicious = 10.
python -m scicheck accuracy    --config configs/paper.yaml --workers 8 --devices cuda:0,cuda:1
python -m scicheck performance --config configs/paper.yaml      # §VI-B; or --sweep runtime|memory
python -m scicheck report      --config configs/paper.yaml      # -> results/paper/report/report.md
```

* **Job files** (`jobs/`): `accuracy.sbatch` is a SLURM array with one task per architecture
  (MLP, CNN, AE); `performance.sbatch` runs the §VI-B sweeps and the report in one exclusive job
  so timings stay clean; `run_all.sh` does the same without a scheduler. All use `--archs` to run
  one architecture per process into a shared results directory.
* `accuracy` is **resumable**: each (arch, size, seed) unit writes `results/<name>/units/<unit>/runs.csv`
  last, and finished units are skipped on restart. Kill it and rerun the same command at any time.
* `--workers N` runs N units in parallel (spawned processes); `--devices` assigns workers to devices
  round-robin. The trained models are tiny (4k–12k params), so a single run uses a sliver of a
  GPU and is bound by per-step Python/kernel-launch overhead. Several workers per GPU is the
  way to use the hardware, and CPU cores are usually the real limit: set `--workers` near the
  core count. It is worth timing one unit with `--devices cpu` against `--devices cuda:0`. At this
  model size the CPU can be as fast.
* **Cost.** `configs/paper.yaml` now uses 100k–500k-parameter models (the paper used 4k–12k).
  Each unit trains 31 models (M0 + 5 honest + 25 malicious) of 25 epochs. At 500k params a CPU
  step costs ≈ 44 ms (MLP), ≈ 250 ms (CNN) and ≈ 26 ms (AE) with 4 threads, so this grid is
  only practical on GPU.
* The output directory stores the settings it was produced with (`config.json`) and refuses to
  mix in results from different settings. Grid axes (seeds, sizes, …) may be extended later.
* `save_snapshots: true` keeps every run's snapshot sequence (`snapshots.npz`, ~1 GB for the full grid)
  and each unit's judge (`judge.pt`), so new detectors and baselines can be scored without retraining.

### Outputs

| file | content |
|---|---|
| `runs.csv` | one row per trained scientist run: ϕ, η̄_s, η̄_v, γ, τ, both check outcomes, decision, final loss |
| `units/*/meta.json` | verifier/judge diagnostics: epochs run, validation history, held-out false-flag rate |
| `report/report.md` | Tables IV–VI, Figure 2, ablation, sensitivity (plus CSV/PNG/PDF of each) |
| `performance/runtime.csv`, `memory.csv` | per-cell stage timings / peak RSS |

## Expected model sizes

`python -m scicheck info` prints the architecture table. The achieved counts span exactly the
**3,922–12,051** range the paper reports, which pins down the shapes: AE `784→h→784` (h=2 gives
3,922) and CNN `conv(3→c)→conv(c→2c)→GAP→1` (c=25 gives 12,051).

| | 4k | 6k | 8k | 10k | 12k | snapshot dim |
|---|---|---|---|---|---|---|
| MLP 8→h→h→1 | 4,003 (h=58) | 5,977 | 7,981 | 10,071 | 11,961 (h=104) | 16 |
| CNN | 3,977 (c=14) | 5,747 | 7,841 | 10,259 | 12,051 (c=25) | 16 |
| AE 784→h→784 | 3,922 (h=2) | 5,491 | 8,629 | 10,198 | 11,767 (h=7) | 15 |

## Choices where the paper is silent or inconsistent

| item | paper | what this code does |
|---|---|---|
| training batch size, Adam lr | not stated | 32, 1e-3 (`training.*`) |
| when the snapshot is taken | "per batch" | after `optimizer.step()`: the gradient is the one actually applied (post-flip), w_t is post-update |
| η̄_v source | Alg. 1 line 9: all of T; §VI-A: training split only | `judge.eta_baseline: train` (switchable to `all`) |
| τ quantile | Alg. 1: "r-th percentile"; §VI-A: (1−r)-quantile | (1−r) = 99th percentile |
| windowing / reassembly | "reassemble per-window scores into a timeline" | stride-1 windows, each timestep = mean over the windows covering it |
| anomaly score | "following [36]" | softmax over the window of −AssDis·T (T=50, layer-averaged) × per-timestep MSE |
| AT early stopping | patience 5 | reference-implementation rule: an epoch improves only if both minimax objectives improve on T_val; best weights restored |
| AT dropout, lr schedule | not stated | 0.0, constant lr (no per-epoch halving) |
| honest runs per severity | 10 per (arch, size, severity) cell | trained once per (arch, size, seed) and counted in every severity cell. Honest runs never see `flip`, so this is equivalent; it is also why the paper's MLP/AE FalsePositive is identical across severities |
| malicious severity in §VI-B | not stated | `performance.severity: 0.3` |
| PetImages class balance | "1,500 images" | 750 / 750 |

## Analyses that need no retraining

The paper's two conjuncts (ϕ ≤ θ, η̄_s ≥ ρ·η̄_v) are recomputed from saved scores, so `report`
also produces:

* **Ablation**: AT-only vs effective-signal-only vs both (pooled FP/FN, mean F1, FN at the mildest flip).
* **Sensitivity**: pooled FP/FN over θ ∈ {0.05…0.5} × ρ ∈ {0.1…1.0}, one heatmap per architecture.
