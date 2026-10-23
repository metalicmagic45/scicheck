import numpy as np
import pandas as pd
import pytest
import torch

from scicheck.anomaly_transformer import AnomalyTransformer, anomaly_scores
from scicheck.attacks import SignFlipAttack, fabricate_tail, poison_data
from scicheck.baselines import fit_baselines
from scicheck.config import DataConfig, JudgeConfig, TrainingConfig, load_config
from scicheck.data import HIGGS_FEATURES, TaskData, load_task_data
from scicheck.judge import replay_fraction, sliding_windows, train_judge, validate, windows_to_timeline
from scicheck.models import ARCHS, build_for_target, count_params, tracked_layers
from scicheck.report import table_vi
from scicheck.snapshots import ETA_COL, SnapshotRecorder, feature_names
from scicheck.training import augment_images, train_run, verifier_steps

SIZES = [4000, 6000, 8000, 10000, 12000]


def test_param_counts_match_paper_range():
    counts = [count_params(build_for_target(a, s)[0]) for a in ARCHS for s in SIZES]
    assert min(counts) == 3922 and max(counts) == 12051


def test_param_counts_hit_large_targets():
    for a in ARCHS:
        for s in (100_000, 300_000, 500_000):
            assert count_params(build_for_target(a, s)[0]) == pytest.approx(s, rel=0.01)


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
    X = rng.normal(size=(400, HIGGS_FEATURES)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    dev = torch.device("cpu")
    tcfg = TrainingConfig(epochs=10, batch_size=32)
    ver = train_run("mlp", 10, X[:200], y[:200], tcfg, seed=1, device=dev)
    jcfg = JudgeConfig(window=8, epochs=2, patience=1)
    judge = train_judge(ver.snapshots, jcfg, rho=0.4, device=dev, seed=0)
    honest = train_run("mlp", 10, X, y, tcfg, seed=2, device=dev)
    bad = train_run("mlp", 10, X, y, tcfg, seed=3, device=dev, attack="signflip", severity=0.5)
    for run in (honest, bad):
        v = validate(judge, run.snapshots, 0.3)
        assert 0 <= v.phi <= 1 and v.eta_s > 0
        assert v.accept == (v.phi <= 0.3 and v.eta_s >= judge.gamma)
    assert judge.gamma == pytest.approx(0.4 * judge.eta_v)
    n_tr = int(len(ver.snapshots) * 0.7)
    assert judge.eta_v == pytest.approx(float(ver.snapshots[:n_tr, ETA_COL].mean()), rel=1e-6)


def test_configs_load():
    for path in ("configs/paper.yaml", "configs/smoke.yaml", "configs/datasplit.yaml", "configs/stepablation.yaml"):
        load_config(path)


def test_datasplit_inherits_paper_design():
    paper, split = load_config("configs/paper.yaml"), load_config("configs/datasplit.yaml")
    assert split.train_frac == 1.0 and split.sizes == [300000] and split.out_dir == "results/datasplit"
    assert split.training == paper.training and split.data == paper.data and split.judge == paper.judge
    assert (split.attacks, split.severities, split.n_honest) == (paper.attacks, paper.severities, paper.n_honest)


def test_paper_design_matches_revision_plan():
    cfg = load_config("configs/paper.yaml")
    assert (cfg.n_honest, cfg.n_malicious, cfg.severities) == (15, 5, [0.1, 0.3, 0.5])
    assert (cfg.performance.n_honest, cfg.performance.n_malicious) == (2, 2)
    assert cfg.verifier_steps == "match" and cfg.validate.replay


def test_table_vi_compares_retraining_with_verification(tmp_path):
    perf = tmp_path / "performance"
    perf.mkdir()
    pd.DataFrame([dict(arch="mlp", target_size=100000, seed=0, retrain_s=60.0, verify_s=70.0),
                  dict(arch="mlp", target_size=200000, seed=0, retrain_s=62.0, verify_s=73.0)]
                 ).to_csv(perf / "runtime_mlp.csv", index=False)
    pd.DataFrame([dict(arch="mlp", target_size=100000, seed=0, peak_mb=900.0),
                  dict(arch="mlp", target_size=200000, seed=0, peak_mb=950.0)]
                 ).to_csv(perf / "memory_mlp.csv", index=False)
    t = table_vi(tmp_path, ["mlp", "cnn"])
    assert list(t.index) == ["100k", "200k"]
    assert t[("Retraining (s)", "MLP")].tolist() == [60.0, 62.0]
    assert t[("SciCheck verification (s)", "MLP")].tolist() == [70.0, 73.0]
    assert t[("Peak memory per model (MB)", "MLP")].tolist() == [900.0, 950.0]


def test_stepablation_differs_from_main_only_in_verifier_steps():
    paper, abl = load_config("configs/paper.yaml"), load_config("configs/stepablation.yaml")
    assert paper.verifier_steps == "match" and abl.verifier_steps == "epochs"
    assert abl.sizes == [300000] and abl.out_dir == "results/stepablation"
    assert abl.train_frac == paper.train_frac == 0.3
    assert (abl.training, abl.data, abl.judge, abl.validate) == (paper.training, paper.data, paper.judge, paper.validate)
    assert (abl.attacks, abl.severities, abl.n_honest, abl.n_malicious) == (
        paper.attacks, paper.severities, paper.n_honest, paper.n_malicious)


def test_task_data_splits_are_disjoint_and_nested():
    X = np.arange(100, dtype=np.float32)[:, None]
    d = TaskData(X=X, y=X[:, 0], n_train=85, n_verifier=26)
    sci, ver, test = d.scientist[0][:, 0], d.verifier[0][:, 0], d.test[0][:, 0]
    assert len(sci) == 85 and len(ver) == 26 and len(test) == 15
    assert set(ver) <= set(sci) and not set(test) & set(sci)


def test_load_task_data_holds_out_test_frac(tmp_path):
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.integers(0, 2, 200), rng.normal(size=(200, HIGGS_FEATURES))])
    (tmp_path / "raw").mkdir()
    np.savetxt(tmp_path / "raw" / "HIGGS.csv", rows, delimiter=",", fmt="%.6f")
    d = load_task_data("higgs", tmp_path, train_frac=0.3, dcfg=DataConfig(higgs_rows=200, test_frac=0.15))
    assert d.X.shape == (200, HIGGS_FEATURES) and d.n_train == 170 and d.n_verifier == 51
    assert set(np.unique(d.y)) <= {0.0, 1.0}


def test_augment_images_flips_and_shifts_per_image():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    out = augment_images(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    again = augment_images(x, torch.Generator().manual_seed(0))
    assert torch.equal(out, again)
    outs = [augment_images(x, torch.Generator().manual_seed(s)) for s in range(10)]
    assert any(not torch.equal(o, x) for o in outs)


def test_train_run_logs_held_out_loss_per_epoch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, HIGGS_FEATURES)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    r = train_run("mlp", 8, X[:200], y[:200], TrainingConfig(epochs=3), seed=0, device=torch.device("cpu"),
                  eval_data=(X[200:], y[200:]))
    assert [h["epoch"] for h in r.history] == [1, 2, 3]
    assert all(np.isfinite(h["eval_loss"]) and 0 <= h["eval_acc"] <= 1 for h in r.history)
    plain = train_run("mlp", 8, X[:200], y[:200], TrainingConfig(epochs=3), seed=0, device=torch.device("cpu"))
    np.testing.assert_array_equal(r.snapshots, plain.snapshots)


def test_verifier_matches_scientist_step_count():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, HIGGS_FEATURES)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    tcfg = TrainingConfig(epochs=3, batch_size=32)
    steps = verifier_steps("match", tcfg, n_scientist_rows=200)
    assert steps == 3 * 7 and verifier_steps("epochs", tcfg, 200) is None
    with pytest.raises(ValueError):
        verifier_steps("same", tcfg, 200)
    sci = train_run("mlp", 8, X, y, tcfg, seed=0, device=torch.device("cpu"))
    ver = train_run("mlp", 8, X[:60], y[:60], tcfg, seed=1, device=torch.device("cpu"), steps=steps)
    assert len(ver.snapshots) == len(sci.snapshots) == steps


def test_replay_check_rejects_fabricated_run():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, HIGGS_FEATURES)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    dev = torch.device("cpu")
    tcfg = TrainingConfig(epochs=4, batch_size=32)
    spe = 7
    ver = train_run("mlp", 8, X[:120], y[:120], tcfg, seed=1, device=dev, steps=4 * spe)
    judge = train_judge(ver.snapshots, JudgeConfig(window=4, epochs=1, patience=1), rho=0.4, device=dev, seed=0)
    honest = train_run("mlp", 8, X, y, tcfg, seed=2, device=dev)
    fake = train_run("mlp", 8, X, y, tcfg, seed=3, device=dev, attack="fabricate", severity=0.5)
    assert replay_fraction(judge.normalize(honest.snapshots), spe, 0.02) <= 0.05
    assert replay_fraction(judge.normalize(fake.snapshots), spe, 0.02) == pytest.approx(14 / 21)
    v = validate(judge, fake.snapshots, theta=1.0, steps_per_epoch=spe)
    assert not v.replay_pass and not v.accept
    assert validate(judge, fake.snapshots, theta=1.0).replay_pass


def test_poison_data_flips_labels_or_inverts_images():
    X = np.zeros((100, 4), dtype=np.float32)
    y = np.zeros(100, dtype=np.float32)
    Xp, yp = poison_data(X, y, "binary", 0.3, seed=0)
    assert yp.sum() == 30 and np.array_equal(Xp, X) and y.sum() == 0
    Xa, ya = poison_data(X, X, "reconstruction", 0.2, seed=0)
    assert (Xa.max(1) == 1).sum() == 20 and ya is Xa and X.sum() == 0


def test_fabricate_tail_replays_last_epoch():
    S = np.arange(12, dtype=np.float32)[:, None]
    out = fabricate_tail(S, n_total=20, steps_per_epoch=4)
    assert out.shape == (20, 1)
    np.testing.assert_array_equal(out[:12, 0], np.arange(12))
    np.testing.assert_array_equal(out[12:, 0], [8, 9, 10, 11, 8, 9, 10, 11])


def test_attacked_runs_keep_full_length():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(320, HIGGS_FEATURES)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    tcfg, dev = TrainingConfig(epochs=4, batch_size=32), torch.device("cpu")
    honest = train_run("mlp", 8, X, y, tcfg, seed=0, device=dev)
    fake = train_run("mlp", 8, X, y, tcfg, seed=0, device=dev, attack="fabricate", severity=0.5)
    data = train_run("mlp", 8, X, y, tcfg, seed=0, device=dev, attack="data", severity=0.3)
    assert len(honest.snapshots) == len(fake.snapshots) == len(data.snapshots) == 40
    np.testing.assert_allclose(fake.snapshots[:20], honest.snapshots[:20], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(fake.snapshots[20:30], fake.snapshots[10:20])
    with pytest.raises(ValueError):
        train_run("mlp", 8, X, y, tcfg, seed=0, device=dev, attack="fabricate", severity=0.9)


def test_baselines_flag_shifted_runs_more():
    rng = np.random.default_rng(0)
    T = rng.normal(size=(600, 16)).astype(np.float32)
    jcfg = JudgeConfig(window=8, epochs=1, patience=1)
    judge = train_judge(T, jcfg, rho=0.4, device=torch.device("cpu"), seed=0)
    bs = fit_baselines(T, judge, jcfg, seed=0)
    assert set(bs) == {"iforest", "ocsvm"}
    same = rng.normal(size=(300, 16)).astype(np.float32)
    shifted = same + 4.0
    for b in bs.values():
        assert b.phi(judge, same) < 0.1 and b.phi(judge, shifted) > 0.9
    broken = same.copy()
    broken[:30] = np.nan
    for b in bs.values():
        assert b.phi(judge, broken) >= 0.1
