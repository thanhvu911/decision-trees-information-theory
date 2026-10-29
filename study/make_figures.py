"""Draw the study's figures from the CSV files in study/results/.

Usage: python study/make_figures.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402


def plain_log_x(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())

ROOT = pathlib.Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"

# reference palette: first three categorical slots (validated all-pairs), text inks, surface
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def fig_spurious_gain():
    t = pd.read_csv(RES / "e1_theory.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, color in [(10, BLUE), (100, ORANGE), (1000, AQUA)]:
        d = t[t.levels == k]
        ax.plot(d.n_rows, d.mean_gain, "o-", color=color, label=f"{k} values, measured")
        ax.plot(d.n_rows, d.theory_gain, "--", color=color, linewidth=1.2, label=f"{k} values, theory")
    ax.axhline(0.1654, color=INK2, linewidth=1)
    ax.text(5500, 0.2, "best real feature: 0.165 bits", color=INK2, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(5e-5, 2)
    ax.set_xlabel("rows in the node")
    ax.set_ylabel("information gain of a random feature (bits)")
    ax.set_title("A meaningless feature gains more when it has more values\nand the node has fewer rows")
    ax.legend(fontsize=8, ncol=1, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "e1_spurious_gain.png", dpi=150); plt.close(fig)


def fig_noise_share():
    t = pd.read_csv(RES / "e1_trees.csv")
    depths = sorted(t.max_depth.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, k in zip(axes, [100, 1000]):
        g = t[t.levels == k].groupby(["criterion", "max_depth"]).rows_reaching_noise_split.mean()
        x = np.arange(len(depths)); w = 0.36
        for i, (crit, color, label) in enumerate([("gain", BLUE, "information gain"),
                                                   ("gain_ratio", ORANGE, "gain ratio")]):
            vals = [g[(crit, d)] for d in depths]
            bars = ax.bar(x + (i - 0.5) * w, [100 * v for v in vals], w - 0.04, color=color,
                          label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, 100 * v + 2, f"{100 * v:.0f}%",
                        ha="center", fontsize=9, color=INK)
        ax.set_xticks(x, [f"max depth {d}" for d in depths])
        ax.set_title(f"random feature with {k:,} values")
        ax.set_ylim(0, 110)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("training rows whose path\ngoes through a noise split (%)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("How much of the tree depends on a meaningless feature (mean of 10 splits)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "e1_noise_splits.png", dpi=150); plt.close(fig)


def fig_pruning():
    t = pd.read_csv(RES / "e2_pruning.csv")
    t = t[t.criterion == "entropy"]
    methods = [("unpruned", BLUE), ("pre-pruned", ORANGE), ("post-pruned", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, ds in zip(axes, ["adult", "breast_cancer", "digits"]):
        d = t[t.dataset == ds]
        for m, color in methods:
            dm = d[d.method == m]
            ax.scatter(dm.leaves, 100 * dm.test_acc, s=36, color=color, edgecolor=SURFACE,
                       linewidth=1, label=m)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,) if ds == "adult" else (1, 2, 5)))
        plain_log_x(ax)
        ax.set_title(ds.replace("_", " "))
        ax.set_xlabel("leaves (log scale)")
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Pruning: tree size vs test accuracy, 10 splits each (entropy; overlapping points hide each other)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "e2_pruning.png", dpi=150); plt.close(fig)


def fig_alpha_path():
    t = pd.read_csv(RES / "e3_alpha_path.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
    allr = t[t.group == "all"].sort_values("leaves")
    ax = axes[0]
    ax.plot(allr.leaves, 100 * allr.cv_accuracy, "o-", color=BLUE, markersize=4,
            label="cross-validated (training split)")
    ax.plot(allr.leaves, 100 * allr.accuracy, "o-", color=ORANGE, markersize=4, label="test")
    best = allr.loc[allr.cv_accuracy.idxmax()]
    ax.axvline(best.leaves, color=INK2, linewidth=1, linestyle=":")
    ax.text(best.leaves * 0.9, 100 * allr.accuracy.max() + 0.4,
            f"chosen by cross-validation: {int(best.leaves)} leaves", color=INK2, fontsize=9, ha="right")
    ax.set_xscale("log"); plain_log_x(ax); ax.set_xlabel("leaves after pruning (log scale)")
    ax.set_ylim(None, 100 * allr.accuracy.max() + 1.2)
    ax.set_ylabel("accuracy (%)"); ax.set_title("Pooled accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for grp, color in [("Female", BLUE), ("Male", ORANGE)]:
        d = t[t.group == grp].sort_values("leaves")
        ax.plot(d.leaves, 100 * d.fnr, "o-", color=color, markersize=4, label=f"{grp.lower()}")
    ax.axvline(best.leaves, color=INK2, linewidth=1, linestyle=":")
    ax.set_xscale("log"); plain_log_x(ax); ax.set_xlabel("leaves after pruning (log scale)")
    ax.set_ylabel("false negative rate (%)")
    ax.set_title("Missed >$50K earners, by sex (test)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "e3_alpha_path.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig_spurious_gain, fig_noise_share, fig_pruning, fig_alpha_path):
        try:
            f(); print("drew", f.__name__)
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e.filename)
