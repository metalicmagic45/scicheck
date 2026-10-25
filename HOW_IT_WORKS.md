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

**3. Build a judge.** The verifier trains a reference model on 30% of the training data, for as
many optimizer steps as the scientist takes on the full training data (`verifier_steps: match`), and
uses its snapshots to fit:
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

With `validate.replay: true` there is a third check. Each step is compared with the step one epoch
earlier (the agreed dataset size and batch size fix the epoch length); ψ is the share of steps whose
distance to it is at most `replay_tol` times the run's median step-to-step distance. Honest training
never repeats itself, while a fabricated run that replays an epoch does, so the run must also have
ψ ≤ `psi_max`. It only catches an exact replay: the fabricated-run attack in the experiments adds
noise to its replay so that ψ stays near zero, which leaves catching it to the other two checks.

**5. Spot-check committed checkpoints.** No check on snapshots alone can stop a forger whose fake
snapshots vary like real ones, so with `validate.spot_segments` > 0 the run must also be backed by
checkpoints, following Proof-of-Learning. Training commits a checkpoint (weights, Adam state, batch
order and augmentation state) for segments of `spot_steps` steps. The verifier picks
`spot_segments` segments at random, re-runs each from its committed start on the committed data, and
measures how far it ends from the committed end checkpoint, relative to how far the segment moved.
The run is rejected if any segment is off by more than δ, which is `spot_margin` times the largest
error when the verifier re-runs its own segments (and at least `spot_floor`). An honest segment
re-runs exactly, up to floating-point nondeterminism on GPU. A sign-flipped or poisoned one does not,
and a fabricated run has no computation behind the checkpoints of its fabricated tail, so it is
caught whenever a sampled segment lands there.

In deployment, storage is bounded per epoch: at the end of each epoch the scientist commits that
epoch's checkpoint hashes to the ledger as one Merkle root, the segments to check are drawn from
randomness that only exists after the commit (such as the next block's hash), and the scientist
keeps those checkpoints and deletes the rest. At most one epoch of checkpoints is on disk at a time.
The experiment picks the segments when each run starts and keeps only their checkpoints in memory,
which is statistically the same because no attack depends on which segments are picked.

The runtime sweep measures what this saves. It takes a real checkpoint from an honest run (weights,
Adam state, augmentation state), measures its size in bytes, and multiplies by the number of
checkpoints each scheme keeps. Keeping every checkpoint until the run is verified, as Proof-of-Learning
does, stores one per segment for the whole run plus every batch index; it is reported with and without
the optimizer state. Committing one epoch at a time stores at most one epoch's checkpoints and batch
indices plus the sampled segments kept so far, and the verifier receives only those sampled segments.
Hashes and Merkle paths are a few kilobytes and are left out. The report shows these totals in its
checkpoint storage table.

## Experiments

Three attacks cover the adversary model, each at severities 0.1, 0.3 and 0.5:
- **sign flip** (tampering): the sign of that fraction of gradient entries is flipped every batch;
- **label poisoning**: that fraction of the training rows, picked at random before training, is
  corrupted: their labels are flipped for the classifiers, and their pixels are inverted for the
  autoencoder, which has no labels;
- **fabricated run**: the scientist trains only the first (1 - severity) of the steps and fills the
  rest of the submitted record by replaying their last genuine epoch, with Gaussian noise added to
  each statistic (`FABRICATE_NOISE` = 0.1 times that statistic's step-to-step standard deviation
  in the genuine steps) so the copy is not exact. For the spot-check it commits its last genuine
  checkpoint for every step it skipped.

SciCheck is compared with two standard anomaly detectors, Isolation Forest and One-Class SVM, fit on
the same verifier snapshots and thresholded the same way (each alone and with the η check). It is
tested on an MLP (HIGGS particle-physics data), a CNN (cats vs dogs) and an autoencoder (MNIST)
at three model sizes. A held-out slice of each dataset, which neither party trains on, records
train vs held-out loss every epoch so overfitting shows up in the report. `accuracy` measures
false positives and false negatives, `performance` measures what verification costs next to retraining (2 clean + 2 dirty runs per size)
and each stage's peak memory, and `report`
builds the tables and figures.

### Run counts

There are 3 architectures × 3 model sizes (100k, 300k and 500k parameters) × 3 seeds, and each
(architecture, size, seed) gets one judge. Each judge is built from one verifier model and checks:
- **15 clean runs**, shared by every attack cell, which give its false-positive rate;
- **3 attack types × 3 severities × 5 dirty runs** = 45 dirty runs, which give its false-negative
  rate.

A table cell is one (attack, severity) pair: its 5 dirty runs against the judge's 15 clean runs.
Every run has its own seed, derived from its architecture, size, role, attack, severity and index.

**Main experiment** (`configs/paper.yaml`, verifier trains on 30% of the training data):

```
3 architectures × 3 sizes × 3 seeds × (3 attack types × 3 severities × 5 dirty + 15 clean) + 27 verifiers
= 27 × 60 + 27
= 1,647 trained models, 27 judges
```

**Data-split control** (`configs/datasplit.yaml`, verifier trains on 100% of the training data,
300k only):

```
3 architectures × 1 size × 3 seeds × (3 attack types × 3 severities × 5 dirty + 15 clean) + 9 verifiers
= 9 × 60 + 9
= 549 trained models, 9 judges
```

The control's clean and dirty runs use the same seeds and data as the main experiment's 300k runs,
so the only thing that changes is the verifier's data.

**Step-matching ablation** (`configs/stepablation.yaml`, 300k only): the main experiment again, with
`verifier_steps: epochs` so the verifier trains for the same number of epochs as the scientist and
therefore takes about 30% as many optimizer steps. It has the same 549 models as the data-split control, with
the same seeds and data as the main experiment's 300k runs, so comparing it with the main run shows
what step matching does and nothing else.

Together the three experiments train 2,745 models.

### Seeds

For each (architecture, size, seed), the **verifier model** is an honest training run whose snapshots
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
3 architectures × 3 sizes × 3 seeds × (3 attack types × 3 severities × 5 dirty + 15 clean) = 1,620 clean + dirty runs
+ 27 judges
+ 27 verifiers
+ 27 baselines
= 1,701 seeds, all different
```

Each baseline seed fits both baseline models, so 27 seeds give 54 baseline models. Every trained
model also splits its seed into sub-seeds for weight initialization, batch order, sign-flip masks,
poisoned rows, augmentation, fabrication noise and which segments get spot-checked, so no two models
start from the same weights or see the same batches.

The data-split control follows the same pattern, 540 + 9 + 9 + 9 = 567 seeds, but reuses the
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
| `anomaly_transformer.py`, `judge.py` | the judge: fitting it and accepting or rejecting a run, and the spot-check re-runs. The Anomaly Transformer is ported from Xu et al. (ICLR 2022), github.com/thuml/Anomaly-Transformer |
| `experiments/`, `worker.py` | accuracy and performance sweeps |
| `storage.py` | checkpoint storage: bytes of a real checkpoint, and the totals each storage scheme keeps |
| `report.py` | tables and figures |
