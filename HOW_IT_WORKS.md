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

**3. Build a judge.** The verifier trains a reference model on 30% of the data and uses its
snapshots to fit:
- an **Anomaly Transformer** that scores each step by how unusual it looks, with threshold τ set
  at the 99th percentile of the verifier's own held-out scores;
- an **effective-signal cutoff** γ = ρ · η̄_v, where η̄_v is the verifier's mean η.

**4. Judge the submission.** The scientist trains on the full data and submits their snapshots.
The run is accepted only if both checks pass:
- ϕ, the fraction of steps scored above τ, is at most θ;
- η̄_s, the run's mean effective signal, is at least γ.

The two checks catch different things: the transformer finds unusual patterns over time, and the
η check finds runs whose gradients are mostly noise.

## Experiments

The attack flips the sign of a random fraction (10-50%) of gradient entries every batch. It is
tested on an MLP (tabular placement data), a CNN (cats vs dogs) and an autoencoder (MNIST)
at five model sizes. `accuracy` measures false positives and false negatives, `performance`
measures runtime and memory, and `report` builds the tables and figures.

## Code

| file | role |
|---|---|
| `snapshots.py` | per-step statistics |
| `training.py`, `models.py`, `attacks.py`, `data.py` | training runs, architectures, sign-flip attack, datasets |
| `anomaly_transformer.py`, `judge.py` | the judge: fitting it and accepting or rejecting a run |
| `experiments/`, `worker.py` | accuracy and performance sweeps |
| `report.py` | tables and figures |
