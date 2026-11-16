"""Fig. 1 of the paper: learning curves, drawn at IEEE single-column width (3.5 in) with 8-pt text.

Usage (from this folder): python make_figure.py   -> ../figures/fig1_learning_curve.png/.pdf
Reads ../results/learning_curve.csv (analysis.py) and ../results/auto_rule_curve.csv (transfer.py auto).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
curve = pd.read_csv(HERE.parent / "results" / "learning_curve.csv")
auto = pd.read_csv(HERE.parent / "results" / "auto_rule_curve.csv").query("model == 'M2auto'")
curve = pd.concat([curve, auto.assign(model="M2-auto")], ignore_index=True)
cc = pd.read_csv(HERE.parent / "results" / "ext_cle_curve_vsM2.csv").query("model == 'M2CC'")
curve = pd.concat([curve, cc.assign(model="M2-CC")], ignore_index=True)
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["M0", "M0b", "M0c", "M1", "M2", "M2-auto", "M2-CC"]
# colorblind-safe (Okabe-Ito) colors, plus distinct markers and line styles (only M2 solid) for grayscale
STYLE = {
    "M0": ("#0072B2", "o", (0, (6, 1.5, 1, 1.5, 1, 1.5))),
    "M0b": ("#56B4E9", "s", "--"),
    "M0c": ("#CC79A7", "v", "-."),
    "M1": ("#D55E00", "^", ":"),
    "M2": ("#009E73", "D", "-"),
    "M2-auto": ("#E69F00", "P", (0, (3, 1))),
    "M2-CC": ("#000000", "X", (0, (1, 1))),
}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "mathtext.fontset": "stix",
    }
)
sizes = sorted(curve.n.unique())
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
for ax, where, title in [
    (axes[0], "test", "(a) Held-out test set"),
    (axes[1], "other_train", "(b) Training patients outside the subset"),
]:
    for i, k in enumerate(MODELS):
        g = curve.query("scored_on == @where and model == @k").sort_values("n")
        col, mk, ls = STYLE[k]
        ax.plot(
            g["n"] * (1 + (i - 2.5) * 0.02),
            g["mean_log_loss"],
            marker=mk,
            color=col,
            ls=ls,
            lw=1.2 if k == "M2" else 0.9,
            ms=3.6,
            markerfacecolor="none" if k in ("M0b", "M2-CC") else col,
            label=k,
        )
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_title(title, loc="left", pad=3)
    ax.set_ylabel("Mean log loss (nats)")
    ax.grid(alpha=0.25, lw=0.4)
axes[1].set_xlabel("Training patients in subset, $n$")
axes[0].legend(
    ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.50), handlelength=2.0, columnspacing=0.8
)
fig.tight_layout(pad=0.3, h_pad=0.6)
fig.savefig(FIG / "fig1_learning_curve.png", dpi=600, facecolor="white")
fig.savefig(FIG / "fig1_learning_curve.pdf", facecolor="white")
print("wrote", FIG / "fig1_learning_curve.png")
