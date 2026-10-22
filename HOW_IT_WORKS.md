# How SciCheck works

One scientist trains a model; another has to check that the training was done honestly under
the agreed setup. Retraining and comparing weights doesn't work, because two honest runs with
different seeds end up with different weights. SciCheck instead asks whether the submitted run
*behaves* like an honest run.

**1. Agree on a blueprint.** Dataset, architecture, hyperparameters and training procedure are
fixed before anyone trains.

**2. Record snapshots.** After every optimizer step, training logs a small vector of statistics:
gradient signal-to-noise, effective signal η, loss, gradient and weight norms, gradient
percentiles, skew and kurtosis, and one gradient norm per layer. A run becomes a time series of
these snapshots. They are summaries, not raw weights, so runs from different seeds are comparable.

**3. Build a judge.** The verifier trains a reference model on 30% of the training data and uses
its snapshots to fit:
- an **Anomaly Transformer** that scores each step by how unusual it looks, with threshold τ set
  at the 99th percentile of its scores on a validation slice of the verifier's own snapshots;
- an **effective-signal cutoff** γ = ρ · η̄_v, where η̄_v is the mean η over the snapshots the
  transformer trains on.

**4. Judge the submission.** The scientist trains on all the training data and submits their
snapshots. (Training data is everything except the held-out test slice, which nobody trains on.)
The run is accepted only if both checks pass:
- ϕ, the fraction of steps scored above τ, is at most θ;
- η̄_s, the run's mean effective signal, is at least γ.

The two checks catch different things: the transformer finds unusual patterns over time, and the
η check finds runs whose gradients are mostly noise.

## Experiments

Three attacks cover the adversary model, each at severities 0.1, 0.3 and 0.5:
- **sign flip** (tampering): the sign of that fraction of gradient entries is flipped every batch;
- **unauthorized data**: that fraction of the training set is replaced with data the blueprint
  doesn't allow (label-flipped rows for the classifiers, pixel-inverted digits for the autoencoder);
- **fabricated run**: the scientist trains only the first (1 - severity) of the steps and fills the
  rest of the submitted record by replaying their last genuine epoch.

SciCheck is compared with two standard anomaly detectors, Isolation Forest and One-Class SVM, fit on
the same verifier snapshots and thresholded the same way (each alone and with the η check). It is
tested on an MLP (HIGGS particle-physics data), a CNN (cats vs dogs) and an autoencoder (MNIST)
at five model sizes. A held-out slice of each dataset, which neither party trains on, records
train vs held-out loss every epoch so overfitting shows up in the report. `accuracy` measures
false positives and false negatives, `performance` measures runtime and memory, and `report`
builds the tables and figures.

### Run counts

There are 3 architectures × 5 model sizes (100k-500k parameters), and each (architecture, size)
gets one judge. Each judge is built from one verifier model and checks:
- **10 clean runs**, shared by every attack cell, which give its false-positive rate;
- **3 attack types × 3 severities × 5 dirty runs** = 45 dirty runs, which give its false-negative
  rate.

A table cell is one (attack, severity) pair: its 5 dirty runs against the judge's 10 clean runs.
Every run has its own seed, derived from its architecture, size, role, attack, severity and index.

**Main experiment** (`configs/paper.yaml`, verifier trains on 30% of the training data):

```
3 architectures × 5 sizes × (3 attack types × 3 severities × 5 dirty + 10 clean) + 15 verifiers
= 15 × 55 + 15
= 840 trained models, 15 judges
```

**Data-split control** (`configs/datasplit.yaml`, verifier trains on 100% of the training data,
300k only):

```
3 architectures × 1 size × (3 attack types × 3 severities × 5 dirty + 10 clean) + 3 verifiers
= 3 × 55 + 3
= 168 trained models, 3 judges
```

The control's clean and dirty runs use the same seeds and data as the main experiment's 300k runs,
so the only thing that changes is the verifier's data. Together the two experiments train 1,008
models.

### Seeds

For each (architecture, size), the **verifier model** is an honest training run whose snapshots
are used to fit the **judge** (the Anomaly Transformer, plus τ and γ calibrated from the same
snapshots) and the **baselines** (Isolation Forest and One-Class SVM, each with its own τ). The
judge and both baselines then score every clean and dirty run.

```
verifier model ──► snapshots ──┬──► judge (AT + τ + γ)
                               └──► baselines (IF, OCSVM, each with its own τ)

each clean/dirty run ──► snapshots ──► scored by the judge and the baselines
```

Each of these gets its own seed. Main experiment:

```
3 architectures × 5 sizes × (3 attack types × 3 severities × 5 dirty + 10 clean) = 825 clean + dirty runs
+ 15 judges
+ 15 verifiers
+ 15 baselines
= 870 seeds, all different
```

Each baseline seed fits both baseline models, so 15 seeds give 30 baseline models. Every trained
model also splits its seed into sub-seeds for weight initialization, batch order, sign-flip masks,
poisoned rows and augmentation, so no two models start from the same weights or see the same
batches.

The data-split control follows the same pattern, 165 + 3 + 3 + 3 = 174 seeds, but reuses the
seeds of the main experiment's 300k units on purpose: the verifier starts from the same seed and
sees different data (100% instead of 30%), so it produces a different judge.

What does not vary is the data split itself. Which rows are held out, and which 30% the verifier
gets, is fixed by `data_seed` for every model, as in the original paper. On GPU the same seed is
not bit-identical between runs (`deterministic: false`), but runs stay statistically equivalent.

## Code

| file | role |
|---|---|
| `snapshots.py` | per-step statistics |
| `training.py`, `models.py`, `attacks.py`, `data.py` | training runs, architectures, the three attacks, datasets |
| `baselines.py` | Isolation Forest and One-Class SVM baselines |
| `anomaly_transformer.py`, `judge.py` | the judge: fitting it and accepting or rejecting a run |
| `experiments/`, `worker.py` | accuracy and performance sweeps |
| `report.py` | tables and figures |
