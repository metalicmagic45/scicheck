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

`download` puts MNIST, PetImages and HIGGS in `data/`. HIGGS comes from UCI as a 2.6 GB zip
(https://archive.ics.uci.edu/static/public/280/higgs.zip); only the first `data.higgs_rows` rows are
read and cached. If you already have `HIGGS.csv.gz` or `HIGGS.csv`, put it in `data/raw/` instead.

## Run

Quick check (a few minutes):

```bash
python -m pytest tests -q
python -m scicheck accuracy --config configs/smoke.yaml
python -m scicheck report   --config configs/smoke.yaml
```

Full experiments, on one GPU machine with no scheduler needed:

```bash
tmux new -s scicheck              # or screen, so the run survives if you disconnect
GPUS=0 jobs/run_all.sh            # GPUS=0,1,2 spreads the three architectures over three GPUs
```

`jobs/run_all.sh` downloads the data, then runs the main experiment (one job per architecture)
alongside the memory sweep (CPU only, the longest step), then the data-split control, then the
runtime sweep on its own so nothing else skews its timings, and finally both reports. Progress is
printed as each step starts and finishes; each step's output is in `logs/<step>.log`. `WORKERS`
(default 4) sets the worker processes per accuracy job. Each step can also be run by hand:

```bash
python -m scicheck accuracy    --config configs/paper.yaml --workers 4 --devices cuda:0
python -m scicheck performance --config configs/paper.yaml
python -m scicheck accuracy    --config configs/datasplit.yaml --workers 4 --devices cuda:0
python -m scicheck report      --config configs/datasplit.yaml
python -m scicheck report      --config configs/paper.yaml
```

`configs/paper.yaml` runs 100k-500k parameter models (batch size 128, 1 seed) against three attacks
(sign flip, unauthorized data, fabricated run) at severities 0.1/0.3/0.5, with 15 honest runs per
judge and 5 malicious runs per (attack, severity) cell. The datasets are large enough for that
size: HIGGS (200k rows) for the MLP, all of PetImages (~25k images, flip/crop augmentation) for the
CNN, and all 60k MNIST digits for the AE, each with 15% held out from both parties to measure
overfitting. [HOW_IT_WORKS.md](HOW_IT_WORKS.md) has the run and seed counts. The original paper used
4k-12k parameters, sign flip only, 5 seeds, 10 + 10 runs per cell, the 10k-row placement table,
1,500 PetImages and 3,000 MNIST digits.

`configs/datasplit.yaml` is the data-split control: the same design at 300k only, with the verifier
trained on all of the scientist's training data (`train_frac: 1.0`) instead of 30%. It writes to
`results/datasplit/`. `configs/stepablation.yaml` is the step-matching ablation: the main design at 300k
with the verifier trained for the same number of epochs as the scientist (`verifier_steps: epochs`)
instead of the same number of optimizer steps, writing to `results/stepablation/`. Every step resumes where it stopped: if the run is interrupted or a step
fails, run it again and finished work is skipped.

## Output

`accuracy` prints one line per finished (architecture, size, seed) in this format (the numbers are
made up):

```
(3/15) [mlp 100000 s0 cuda:0] done in 2400s | honest rejected 1/10 | malicious rejected signflip 14/15, data 9/15, fabricate 12/15
```

Everything is written to the config's `out_dir` (e.g. `results/paper/`):

| path | contents |
|---|---|
| `runs.csv` | one row per scientist run: role, attack, severity, phi, eta_s, psi, each check's result, accept, the baselines' phi_iforest / phi_ocsvm, final held-out loss |
| `units/<arch>_<size>_s<seed>/` | that unit's runs, judge (`judge.pt`), snapshots, training diagnostics (`meta.json`) |
| `units/.../history.csv` | train and held-out loss (and accuracy) after every epoch of every run |
| `performance/` | per (arch, size): retraining time vs SciCheck verification time (verifier, judge fit, scoring one submission), and peak memory of each stage |
| `report/report.md` | Tables IV-VI, Figure 2, baselines vs ablation, θ/ρ sensitivity, overfitting table and gap plot, plus CSV/PNG of each |

For reference, the original paper's 4k-12k sign-flip runs: the MLP and CNN rejected nearly all
malicious runs at flip ≥ 0.3 and missed many at 0.1, and the autoencoder rejected most honest runs as
well. The other two attacks and the baselines have no earlier results. The smoke config is too small
for meaningful numbers; it only checks that everything runs end to end.
