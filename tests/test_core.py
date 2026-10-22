import numpy as np
import pytest
import torch

from scicheck.anomaly_transformer import AnomalyTransformer, anomaly_scores
from scicheck.attacks import SignFlipAttack
from scicheck.config import JudgeConfig, TrainingConfig, load_config
from scicheck.judge import sliding_windows, train_judge, validate, windows_to_timeline
from scicheck.models import ARCHS, build_for_target, count_params, tracked_layers
from scicheck.snapshots import ETA_COL, SnapshotRecorder, feature_names
from scicheck.training import train_run

SIZES = [4000, 6000, 8000, 10000, 12000]


def test_param_counts_match_paper_range():
    counts = [count_params(build_for_target(a, s)[0]) for a in ARCHS for s in SIZES]
    assert min(counts) == 3922 and max(counts) == 12051


@pytest.mark.parametrize("arch,dim", [("mlp", 16), ("cnn", 16), ("ae", 15)])
def test_snapshot_dim_matches_table_iii(arch, dim):
    m, _ = build_for_target(arch, 4000)
    assert len(feature_names(len(tracked_layers(m)))) == dim


def test_sign_flip_scales_effective_signal():
    torch.manual_seed(0)
    g = torch.randn(200_000) * 0.1 + 0.05
    eta = lambda v: float(v.mean() ** 2 / (v.mean() ** 2 + v.var(unbiased=False)))
    p = torch.nn.Parameter(torch.zeros_like(g))
    for flip in (0.1, 0.2, 0.5):
        p.grad = g.clone()
        SignFlipAttack(flip, seed=1, device=torch.device("cpu"))([p])
        assert eta(p.grad) / eta(g) == pytest.approx((1 - 2 * flip) ** 2, abs=0.02)


def test_attack_flips_requested_fraction():
    p = torch.nn.Parameter(torch.zeros(100_000))
    p.grad = torch.ones(100_000)
    SignFlipAttack(0.3, seed=0, device=torch.device("cpu"))([p])
    assert float((p.grad < 0).float().mean()) == pytest.approx(0.3, abs=0.01)


def _reference_snapshot(model, layers, loss, prev_w, prev_loss):
    ps = list(model.parameters())
    g = torch.cat([p.grad.reshape(-1) for p in ps]).double()
    w = torch.cat([p.detach().reshape(-1) for p in ps]).double()
    m, s = g.mean(), g.std(unbiased=False)
    z = (g - m) / s
    q = torch.quantile(g.abs(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.double))
    row = [m.abs() / s, loss - prev_loss if prev_loss is not None else 0.0, m**2 / (m**2 + s**2),
           g.norm(), (w - prev_w).norm(), loss, w.norm(), (g > 0).double().mean(), q[0], q[1], q[2],
           (z**3).mean(), (z**4).mean() - 3]
    row += [torch.cat([p.grad.reshape(-1) for p in l.parameters()]).norm() for l in layers]
    return np.array([float(v) for v in row]), w


def test_chunked_recorder_matches_naive_reference():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    layers = tracked_layers(model)
    rec = SnapshotRecorder(model, layers, chunk=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    attack = SignFlipAttack(0.3, seed=0, device=torch.device("cpu"))
    prev_w = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double()
    prev_loss, expected = None, []
    for _ in range(5):
        x, y = torch.randn(32, 4), torch.randn(32, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        attack(list(model.parameters()))
        opt.step()
        rec.record(loss)
        row, prev_w = _reference_snapshot(model, layers, loss.item(), prev_w, prev_loss)
        prev_loss = loss.item()
        expected.append(row)
    S = rec.to_numpy()
    assert S.shape == (5, len(feature_names(2)))
    np.testing.assert_allclose(S, np.array(expected), rtol=2e-4, atol=1e-6)


def test_timeline_reassembly_averages_overlaps():
    n, W = 7, 3
    scores = np.tile(np.arange(n - W + 1, dtype=float)[:, None], (1, W))
    tl = windows_to_timeline(scores, n)
    assert tl[0] == 0 and tl[-1] == n - W
    assert tl[2] == pytest.approx(np.mean([0, 1, 2]))
    assert sliding_windows(np.zeros((n, 5)), W).shape == (n - W + 1, W, 5)


def test_anomaly_transformer_shapes():
    m = AnomalyTransformer(win_size=30, c_in=16)
    x = torch.randn(4, 30, 16)
    out, series, prior = m(x)
    assert out.shape == x.shape and len(series) == len(prior) == 2
    assert series[0].shape == (4, 4, 30, 30)
    assert anomaly_scores(m, x, 50.0).shape == (4, 30)


def test_end_to_end_on_synthetic_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    dev = torch.device("cpu")
    tcfg = TrainingConfig(epochs=10, batch_size=32)
    ver = train_run("mlp", 10, X[:200], y[:200], tcfg, seed=1, device=dev)
    jcfg = JudgeConfig(window=8, epochs=2, patience=1)
    judge = train_judge(ver.snapshots, jcfg, rho=0.4, device=dev, seed=0)
    honest = train_run("mlp", 10, X, y, tcfg, seed=2, device=dev)
    bad = train_run("mlp", 10, X, y, tcfg, seed=3, device=dev, flip=0.5)
    for run in (honest, bad):
        v = validate(judge, run.snapshots, 0.3)
        assert 0 <= v.phi <= 1 and v.eta_s > 0
        assert v.accept == (v.phi <= 0.3 and v.eta_s >= judge.gamma)
    assert judge.gamma == pytest.approx(0.4 * judge.eta_v)
    n_tr = int(len(ver.snapshots) * 0.7)
    assert judge.eta_v == pytest.approx(float(ver.snapshots[:n_tr, ETA_COL].mean()), rel=1e-6)


def test_configs_load():
    for path in ("configs/paper.yaml", "configs/smoke.yaml"):
        load_config(path)
