from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .experiments.accuracy import collect_runs

ARCH_LABEL = {"mlp": "MLP", "cnn": "CNN", "ae": "AE"}
SEVERITY_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ATTACK_COLORS = {"signflip": "#2a78d6", "data": "#d97a1f", "fabricate": "#3a9a5b"}
ATTACK_LABEL = {"signflip": "Sign flip", "data": "Unauthorized data", "fabricate": "Fabricated run"}
PHI_COL = {"at": "phi", "iforest": "phi_iforest", "ocsvm": "phi_ocsvm"}
CHECKS = {
    "at+eta+replay": "SciCheck + replay",
    "replay": "Replay only",
    "at+eta": "SciCheck (AT + η)",
    "at": "AT only",
    "eta": "η only",
    "iforest": "Isolation Forest",
    "iforest+eta": "Isolation Forest + η",
    "ocsvm": "One-Class SVM",
    "ocsvm+eta": "One-Class SVM + η",
}
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED, GRID = "#1f1f1e", "#5f5f5c", "#e4e3df"
THETA_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
RHO_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def size_label(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def decide(df: pd.DataFrame, theta: float, rho: float, variant: str = "at+eta") -> pd.Series:
    """Accept where every check named in `variant` ("at+eta", "iforest", "replay", ...) passes."""
    passed = pd.Series(True, index=df.index)
    for check in variant.split("+"):
        if check == "eta":
            passed &= df.eta_s >= rho * df.eta_v
        elif check == "replay":
            passed &= df.replay_pass.astype(bool)
        else:
            passed &= df[PHI_COL[check]] <= theta
    return passed


def has_checks(cells: pd.DataFrame, variant: str) -> bool:
    cols = {"eta": "eta_s", "replay": "replay_pass", **PHI_COL}
    return all(cols[c] in cells for c in variant.split("+"))


def default_variant(cells: pd.DataFrame) -> str:
    return "at+eta+replay" if "replay_pass" in cells else "at+eta"


def expand_cells(df: pd.DataFrame) -> pd.DataFrame:
    mal, honest = df[df.role == "malicious"], df[df.role == "honest"]
    parts = [mal.assign(cell_attack=mal.attack, cell_sev=mal.severity)]
    parts += [honest.assign(cell_attack=a, cell_sev=s)
              for a, s in mal[["attack", "severity"]].drop_duplicates().itertuples(index=False)]
    return pd.concat(parts, ignore_index=True)


def _attack_order(cells: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(cells.cell_attack))


def counts(g: pd.DataFrame, accept: str = "accept") -> pd.Series:
    mal, rej = g.role == "malicious", ~g[accept].astype(bool)
    return pd.Series({"TP": (mal & rej).sum(), "FP": (~mal & rej).sum(),
                      "FN": (mal & ~rej).sum(), "TN": (~mal & ~rej).sum()})


def prf(c: pd.Series) -> pd.Series:
    p = c.TP / (c.TP + c.FP) if c.TP + c.FP else 0.0
    r = c.TP / (c.TP + c.FN) if c.TP + c.FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return pd.Series({"P": p, "R": r, "F1": f1})


def rates(c: pd.Series) -> pd.Series:
    fp = c.FP / (c.FP + c.TN) if c.FP + c.TN else np.nan
    fn = c.FN / (c.FN + c.TP) if c.FN + c.TP else np.nan
    return pd.Series({"FalsePositive": fp, "FalseNegative": fn})


def cell_prf(cells: pd.DataFrame, accept: str = "accept") -> pd.DataFrame:
    keys = ["arch", "target_size", "cell_attack", "cell_sev", "seed"]
    return cells.groupby(keys).apply(lambda g: prf(counts(g, accept)), include_groups=False).reset_index()


def _arch_blocks(fn, archs) -> pd.DataFrame:
    blocks = {ARCH_LABEL.get(a, a): fn(a) for a in archs}
    return pd.concat(blocks, axis=1)


def _breakdown(d: pd.DataFrame, stat) -> pd.DataFrame:
    order = _attack_order(d)
    by_size = d.groupby("target_size").apply(stat, include_groups=False)
    by_size.index = [f"size {size_label(s)}" for s in by_size.index]
    by_attack = d.groupby("cell_attack").apply(stat, include_groups=False).reindex(order)
    by_attack.index = [ATTACK_LABEL.get(a, a) for a in by_attack.index]
    by_sev = d.groupby(["cell_attack", "cell_sev"]).apply(stat, include_groups=False)
    by_sev = by_sev.reindex([i for a in order for i in by_sev.index if i[0] == a])
    by_sev.index = [f"{ATTACK_LABEL.get(a, a)} {s:g}" for a, s in by_sev.index]
    overall = stat(d).to_frame("Overall Avg").T
    return pd.concat([by_size, by_attack, by_sev, overall])


def table_iv(cells: pd.DataFrame, archs) -> pd.DataFrame:
    per_cell = cell_prf(cells)
    mean_prf = lambda g: g[["P", "R", "F1"]].mean()
    return _arch_blocks(lambda a: _breakdown(per_cell[per_cell.arch == a], mean_prf), archs)


def table_v(cells: pd.DataFrame, archs) -> pd.DataFrame:
    return _arch_blocks(lambda a: _breakdown(cells[cells.arch == a], lambda g: rates(counts(g))), archs)


def comparison(cells: pd.DataFrame, theta: float, rho: float, archs) -> pd.DataFrame:
    rows = []
    checks = [v for v in CHECKS if has_checks(cells, v)]
    for a in archs:
        d = cells[cells.arch == a].copy()
        for v in checks:
            d["_acc"] = decide(d, theta, rho, v)
            row = {"arch": ARCH_LABEL.get(a, a), "check": CHECKS[v], **rates(counts(d, "_acc")).to_dict()}
            for atk, g in d[d.role == "malicious"].groupby("cell_attack", sort=False):
                row[f"FalseNegative {ATTACK_LABEL.get(atk, atk)}"] = rates(counts(g, "_acc")).FalseNegative
            row["mean F1"] = cell_prf(d, "_acc").F1.mean()
            rows.append(row)
    return pd.DataFrame(rows).set_index(["arch", "check"])


def sensitivity(cells: pd.DataFrame, archs) -> pd.DataFrame:
    rows = []
    for a in archs:
        d = cells[cells.arch == a].copy()
        for th in THETA_GRID:
            for rh in RHO_GRID:
                d["_acc"] = decide(d, th, rh, default_variant(d))
                rows.append({"arch": a, "theta": th, "rho": rh, **rates(counts(d, "_acc")).to_dict()})
    return pd.DataFrame(rows)


TABLE_VI = (
    ("runtime*.csv", "retrain_s", "Retraining (s)"),
    ("runtime*.csv", "verify_s", "SciCheck verification (s)"),
    ("memory*.csv", "peak_mb", "Peak memory per model (MB)"),
)


def table_vi(out: Path, archs) -> pd.DataFrame | None:
    perf = out / "performance"
    frames = []
    for pattern, col, label in TABLE_VI:
        files = sorted(perf.glob(pattern))
        d = pd.concat([pd.read_csv(f) for f in files]) if files else None
        if d is not None and col in d:
            d = d.groupby(["target_size", "arch"])[col].mean().unstack("arch")
            d = d[[a for a in archs if a in d.columns]]
            d.columns = pd.MultiIndex.from_product([[label], [ARCH_LABEL.get(c, c) for c in d.columns]])
            frames.append(d)
    if not frames:
        return None
    t = pd.concat(frames, axis=1)
    t.index = [size_label(s) for s in t.index]
    return t


def load_histories(out: Path) -> pd.DataFrame | None:
    frames = []
    for f in sorted((out / "units").glob("*/history.csv")):
        meta = json.loads((f.parent / "meta.json").read_text())
        h = pd.read_csv(f)
        frames.append(h[h.role.isin(["verifier", "honest"])].assign(arch=meta["arch"],
                                                                       target_size=meta["target_size"]))
    return pd.concat(frames, ignore_index=True) if frames else None


def overfitting(hist: pd.DataFrame, archs) -> pd.DataFrame:
    last = hist[hist.epoch == hist.groupby(["arch", "target_size", "role"]).epoch.transform("max")]
    cols = ["train_loss", "eval_loss"] + (["eval_acc"] if "eval_acc" in last else [])
    t = last.groupby(["arch", "target_size", "role"])[cols].mean()
    t.insert(2, "gap", t.eval_loss - t.train_loss)
    t = t.rename(columns={"train_loss": "train loss", "eval_loss": "held-out loss", "eval_acc": "held-out acc"})
    t = t.reindex([i for a in archs for i in t.index if i[0] == a])
    t.index = [(ARCH_LABEL.get(a, a), size_label(s), r) for a, s, r in t.index]
    return t


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_2(cells: pd.DataFrame, archs, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_cell = cell_prf(cells)
    sizes = sorted(per_cell.target_size.unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(3.4 * len(archs), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, archs):
        d = per_cell[per_cell.arch == a]
        x = np.arange(len(sizes))
        for atk in _attack_order(cells):
            y = d[d.cell_attack == atk].groupby("target_size").F1.mean().reindex(sizes)
            ax.plot(x, y, color=ATTACK_COLORS.get(atk, INK_MUTED), lw=2, marker="o", ms=4,
                    label=ATTACK_LABEL.get(atk, atk))
        avg = d.groupby("target_size").F1.mean().reindex(sizes)
        ax.plot(x, avg, color=INK, lw=2, ls="--", marker="*", ms=8, label="average")
        ax.set_xticks(x, [size_label(s) for s in sizes])
        ax.set_ylim(0, 1.02)
        ax.set_title(ARCH_LABEL.get(a, a), fontsize=10, color=INK)
        ax.set_xlabel("Model size", fontsize=9, color=INK_MUTED)
        _style(ax)
    axes[0].set_ylabel("F1", fontsize=9, color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def figure_sensitivity(sens: pd.DataFrame, archs, theta: float, rho: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    metrics = ["FalsePositive", "FalseNegative"]
    fig, axes = plt.subplots(len(metrics), len(archs), figsize=(3.6 * len(archs), 5.6), squeeze=False)
    for j, a in enumerate(archs):
        d = sens[sens.arch == a]
        for i, m in enumerate(metrics):
            ax = axes[i, j]
            grid = d.pivot(index="theta", columns="rho", values=m).reindex(index=THETA_GRID, columns=RHO_GRID)
            ax.imshow(grid.values, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
            for (yi, xi), v in np.ndenumerate(grid.values):
                ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 else INK)
            ax.set_xticks(range(len(RHO_GRID)), [f"{r:g}" for r in RHO_GRID], fontsize=7)
            ax.set_yticks(range(len(THETA_GRID)), [f"{t:g}" for t in THETA_GRID], fontsize=7)
            if theta in THETA_GRID and rho in RHO_GRID:
                ax.add_patch(plt.Rectangle((RHO_GRID.index(rho) - 0.5, THETA_GRID.index(theta) - 0.5), 1, 1,
                                           fill=False, ec=INK, lw=1.5))
            ax.set_title(f"{ARCH_LABEL.get(a, a)} - {m}", fontsize=9, color=INK)
            ax.set_xlabel("ρ", fontsize=8, color=INK_MUTED)
            ax.set_ylabel("θ", fontsize=8, color=INK_MUTED)
            ax.tick_params(colors=INK_MUTED, length=0)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_gap(hist: pd.DataFrame, archs, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = hist[hist.role == "honest"]
    archs = [a for a in archs if a in set(h.arch)]
    sizes = sorted(h.target_size.unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(3.4 * len(archs), 2.8), squeeze=False)
    for ax, a in zip(axes[0], archs):
        d = h[h.arch == a]
        for i, s in enumerate(sizes):
            g = d[d.target_size == s].groupby("epoch")[["train_loss", "eval_loss"]].mean()
            ax.plot(g.index, g.eval_loss - g.train_loss, color=SEVERITY_COLORS[i % len(SEVERITY_COLORS)],
                    lw=2, label=size_label(s))
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        ax.set_title(ARCH_LABEL.get(a, a), fontsize=10, color=INK)
        ax.set_xlabel("Epoch", fontsize=9, color=INK_MUTED)
        _style(ax)
    axes[0, 0].set_ylabel("Held-out minus train loss", fontsize=9, color=INK_MUTED)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def to_markdown(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = [" ".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in df.columns]
    idx = [" / ".join(str(x) for x in i) if isinstance(i, tuple) else str(i) for i in df.index]
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else str(v)
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += ["| " + i + " | " + " | ".join(cell(v) for v in row) + " |" for i, row in zip(idx, df.values)]
    return "\n".join(lines)


def build_report(cfg: Config) -> Path:
    out = cfg.out_path
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    theta, rho = cfg.validate.theta, cfg.validate.rho
    md = [f"# SciCheck results: `{cfg.name}`", "",
          f"θ = {theta}, ρ = {rho}, r = {cfg.judge.r}, train_frac = {cfg.train_frac}, "
          f"replay check {'on' if cfg.validate.replay else 'off'}. "
          "Positive class = malicious (rejected).", ""]

    try:
        runs = collect_runs(out)
    except FileNotFoundError:
        runs = None
    if runs is not None:
        archs = [a for a in cfg.architectures if a in set(runs.arch)]
        cells = expand_cells(runs)
        n_units = runs.groupby(["arch", "target_size", "seed"]).ngroups
        md += [f"{len(runs)} trained scientist runs from {n_units} units; "
               f"{len(cells)} scored runs after giving each (attack, severity) cell its honest runs.", ""]
        t4, t5 = table_iv(cells, archs), table_v(cells, archs)
        comp, sens = comparison(cells, theta, rho, archs), sensitivity(cells, archs)
        t4.to_csv(rep / "table_iv_prf.csv")
        t5.to_csv(rep / "table_v_fp_fn.csv")
        comp.to_csv(rep / "baselines_ablation.csv")
        sens.to_csv(rep / "sensitivity_theta_rho.csv", index=False)
        figure_2(cells, archs, rep / "fig2_f1_by_size.png")
        figure_sensitivity(sens, archs, theta, rho, rep / "sensitivity_theta_rho.png")
        md += ["## Table IV - precision / recall / F1", "", to_markdown(t4), "",
               "## Table V - FalsePositive / FalseNegative (pooled counts)", "", to_markdown(t5), "",
               "## Figure 2 - F1 by model size and attack", "", "![fig2](fig2_f1_by_size.png)", "",
               "## Baselines and ablation (pooled over all cells)", "",
               "Every check uses the same verifier snapshots, τ at the (1 - r)-quantile of its validation "
               "scores, and ϕ ≤ θ. Isolation Forest and One-Class SVM score each snapshot on its own. "
               "The replay check rejects runs whose steps repeat the step one epoch earlier "
               f"(ψ > {cfg.validate.psi_max}).", "",
               to_markdown(comp), "",
               "## Sensitivity of FalsePositive / FalseNegative to (θ, ρ)", "",
               "Full grid in `sensitivity_theta_rho.csv`; the paper's operating point is outlined.", "",
               "![sensitivity](sensitivity_theta_rho.png)", ""]

    hist = load_histories(out)
    if hist is not None:
        ov = overfitting(hist, cfg.architectures)
        ov.to_csv(rep / "overfitting.csv")
        figure_gap(hist, cfg.architectures, rep / "overfitting_gap.png")
        md += ["## Overfitting - final-epoch train vs held-out loss", "",
               f"Held-out set: {cfg.data.test_frac:.0%} of each dataset, never trained on. Honest runs and the "
               "verifier only. A gap that grows over epochs means the model is memorizing.", "",
               to_markdown(ov, "{:.4f}"), "", "![gap](overfitting_gap.png)", ""]

    t6 = table_vi(out, cfg.architectures)
    if t6 is not None:
        t6.to_csv(rep / "table_vi_runtime_memory.csv")
        md += ["## Table VI - cost of verification vs retraining", "",
               "Retraining: one honest run of the blueprint on the full training data, the cost of checking a "
               "run by training it again. SciCheck verification: the verifier's reference run "
               f"({cfg.train_frac:.0%} of the data, verifier_steps = {cfg.verifier_steps}), fitting the judge, and "
               "scoring one submission. Peak memory: the most any single stage (verifier, one scientist, or judge) "
               "needs, each measured in its own process on CPU. Raw per-stage numbers are in `performance/`.", "",
               to_markdown(t6, "{:,.2f}"), ""]

    (rep / "report.md").write_text("\n".join(md), encoding="utf-8")
    return rep / "report.md"
