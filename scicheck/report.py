"""Builds the paper's evaluation artifacts from saved per-run scores.

  Table IV : mean precision / recall / F1 per (arch, size) and (arch, severity) cell
  Table V  : pooled FalsePositive / FalseNegative from summed confusion counts
  Fig. 2   : F1 by model size, one line per severity plus the size-only average
  Table VI : runtime / memory per (arch, size), from the performance sweeps

Because every run's ϕ, η̄_s and the judge's η̄_v are saved, two further analyses need no
retraining and are produced alongside:
  Ablation    : AT check alone vs effective-signal check alone vs both (§V-D predicate)
  Sensitivity : pooled FalsePositive / FalseNegative over a (θ, ρ) grid

Positive class = malicious; a run is predicted positive when it is rejected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .experiments.accuracy import collect_runs

ARCH_LABEL = {"mlp": "MLP", "cnn": "CNN", "ae": "AE"}
SEVERITY_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, light->dark
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED, GRID = "#1f1f1e", "#5f5f5c", "#e4e3df"
THETA_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
RHO_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def size_label(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


# ----------------------------------------------------------------------------- decisions & metrics

def decide(df: pd.DataFrame, theta: float, rho: float, variant: str = "both") -> pd.Series:
    """Re-apply the acceptance predicate at any (θ, ρ) from saved scores."""
    at_pass = df.phi <= theta
    eta_pass = df.eta_s >= rho * df.eta_v
    return {"both": at_pass & eta_pass, "at": at_pass, "eta": eta_pass}[variant]


def expand_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Honest runs are shared by every severity of a unit; give each severity cell its copy."""
    severities = sorted(df.loc[df.role == "malicious", "flip"].unique())
    honest = df[df.role == "honest"]
    parts = [df[df.role == "malicious"].assign(cell_flip=df.flip)]
    parts += [honest.assign(cell_flip=f) for f in severities]
    return pd.concat(parts, ignore_index=True)


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
    """P/R/F1 per (arch, size, severity, seed) cell -- the unit Table IV averages over."""
    keys = ["arch", "target_size", "cell_flip", "seed"]
    return cells.groupby(keys).apply(lambda g: prf(counts(g, accept)), include_groups=False).reset_index()


# ----------------------------------------------------------------------------- tables

def _arch_blocks(fn, archs) -> pd.DataFrame:
    blocks = {ARCH_LABEL.get(a, a): fn(a) for a in archs}
    return pd.concat(blocks, axis=1)


def table_iv(cells: pd.DataFrame, archs) -> pd.DataFrame:
    per_cell = cell_prf(cells)

    def block(a):
        d = per_cell[per_cell.arch == a]
        by_size = d.groupby("target_size")[["P", "R", "F1"]].mean()
        by_size.index = [f"size {size_label(s)}" for s in by_size.index]
        by_sev = d.groupby("cell_flip")[["P", "R", "F1"]].mean()
        by_sev.index = [f"flip {f:g}" for f in by_sev.index]
        overall = d[["P", "R", "F1"]].mean().to_frame("Overall Avg").T
        return pd.concat([by_size, by_sev, overall])

    return _arch_blocks(block, archs)


def table_v(cells: pd.DataFrame, archs) -> pd.DataFrame:
    def block(a):
        d = cells[cells.arch == a]
        by_size = d.groupby("target_size").apply(lambda g: rates(counts(g)), include_groups=False)
        by_size.index = [f"size {size_label(s)}" for s in by_size.index]
        by_sev = d.groupby("cell_flip").apply(lambda g: rates(counts(g)), include_groups=False)
        by_sev.index = [f"flip {f:g}" for f in by_sev.index]
        overall = rates(counts(d)).to_frame("Overall Avg").T
        return pd.concat([by_size, by_sev, overall])

    return _arch_blocks(block, archs)


def ablation(cells: pd.DataFrame, theta: float, rho: float, archs) -> pd.DataFrame:
    rows = []
    names = {"at": "AT only (ϕ ≤ θ)", "eta": "Effective signal only (η̄s ≥ γ)", "both": "Both (SciCheck)"}
    for a in archs:
        d = cells[cells.arch == a].copy()
        for v, name in names.items():
            d["_acc"] = decide(d, theta, rho, v)
            c = counts(d, "_acc")
            f1 = cell_prf(d, "_acc").F1.mean()
            low = d[d.cell_flip == d.cell_flip.min()]
            rows.append({"arch": ARCH_LABEL.get(a, a), "check": name, **rates(c).to_dict(),
                         f"FalseNegative @ flip {low.cell_flip.min():g}": rates(counts(low, "_acc")).FalseNegative,
                         "mean F1": f1})
    return pd.DataFrame(rows).set_index(["arch", "check"])


def sensitivity(cells: pd.DataFrame, archs) -> pd.DataFrame:
    rows = []
    for a in archs:
        d = cells[cells.arch == a].copy()
        for th in THETA_GRID:
            for rh in RHO_GRID:
                d["_acc"] = decide(d, th, rh)
                rows.append({"arch": a, "theta": th, "rho": rh, **rates(counts(d, "_acc")).to_dict()})
    return pd.DataFrame(rows)


def table_vi(out: Path, archs) -> pd.DataFrame | None:
    perf = out / "performance"
    frames = []
    for pattern, col, label in (("runtime*.csv", "total_s", "Total Time (s)"), ("memory*.csv", "total_mb", "Total Memory (MB)")):
        files = sorted(perf.glob(pattern))  # runtime_<arch>.csv (or a legacy runtime.csv)
        if files:
            d = pd.concat([pd.read_csv(f) for f in files]).groupby(["target_size", "arch"])[col].mean().unstack("arch")
            d = d[[a for a in archs if a in d.columns]]
            d.columns = pd.MultiIndex.from_product([[label], [ARCH_LABEL.get(c, c) for c in d.columns]])
            frames.append(d)
    if not frames:
        return None
    t = pd.concat(frames, axis=1)
    t.index = [size_label(s) for s in t.index]
    return t


# ----------------------------------------------------------------------------- figures

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
    flips = sorted(per_cell.cell_flip.unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(3.4 * len(archs), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, archs):
        d = per_cell[per_cell.arch == a]
        x = np.arange(len(sizes))
        for i, f in enumerate(flips):
            y = d[d.cell_flip == f].groupby("target_size").F1.mean().reindex(sizes)
            ax.plot(x, y, color=SEVERITY_COLORS[i % len(SEVERITY_COLORS)], lw=2, marker="o", ms=4,
                    label=f"flip {f:g}")
        avg = d.groupby("target_size").F1.mean().reindex(sizes)
        ax.plot(x, avg, color=INK, lw=2, ls="--", marker="*", ms=8, label="average")
        ax.set_xticks(x, [size_label(s) for s in sizes])
        ax.set_ylim(0, 1.02)
        ax.set_title(ARCH_LABEL.get(a, a), fontsize=10, color=INK)
        ax.set_xlabel("Model size", fontsize=9, color=INK_MUTED)
        _style(ax)
    axes[0].set_ylabel("F1", fontsize=9, color=INK_MUTED)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
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
            if theta in THETA_GRID and rho in RHO_GRID:  # outline the paper's operating point
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


# ----------------------------------------------------------------------------- markdown

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
          f"θ = {theta}, ρ = {rho}, r = {cfg.judge.r}, train_frac = {cfg.train_frac}. "
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
               f"{len(cells)} scored runs after giving each severity cell its honest runs.", ""]
        t4, t5 = table_iv(cells, archs), table_v(cells, archs)
        abl, sens = ablation(cells, theta, rho, archs), sensitivity(cells, archs)
        t4.to_csv(rep / "table_iv_prf.csv")
        t5.to_csv(rep / "table_v_fp_fn.csv")
        abl.to_csv(rep / "ablation.csv")
        sens.to_csv(rep / "sensitivity_theta_rho.csv", index=False)
        figure_2(cells, archs, rep / "fig2_f1_by_size.png")
        figure_sensitivity(sens, archs, theta, rho, rep / "sensitivity_theta_rho.png")
        md += ["## Table IV - precision / recall / F1", "", to_markdown(t4), "",
               "## Table V - FalsePositive / FalseNegative (pooled counts)", "", to_markdown(t5), "",
               "## Figure 2 - F1 by model size", "", "![fig2](fig2_f1_by_size.png)", "",
               "## Ablation - which conjunct does the work (pooled over all cells)", "", to_markdown(abl), "",
               "## Sensitivity of FalsePositive / FalseNegative to (θ, ρ)", "",
               "Full grid in `sensitivity_theta_rho.csv`; the paper's operating point is outlined.", "",
               "![sensitivity](sensitivity_theta_rho.png)", ""]

    t6 = table_vi(out, cfg.architectures)
    if t6 is not None:
        t6.to_csv(rep / "table_vi_runtime_memory.csv")
        md += ["## Table VI - total runtime and memory", "", to_markdown(t6, "{:,.2f}"), ""]

    (rep / "report.md").write_text("\n".join(md), encoding="utf-8")
    return rep / "report.md"
