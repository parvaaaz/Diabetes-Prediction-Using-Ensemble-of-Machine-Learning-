"""Render paper figures from existing result CSVs.

Usage (cwd = pipeline/):  python src/make_figures.py [dataset2|dataset1] [all|fig5|fig6|fig7|fig6b]
Reads:   results/{DS}_table5..8.csv, {DS}_emlm_members.csv, {DS}_table9.csv, {DS}_emlm_folds.csv
Writes:  figures/{DS}_fig5.png, {DS}_fig6a.png, {DS}_fig6b.png, fig7_acc_comparison.png

Fig.6b (ROC of optimal EMLM) uses the one-vs-rest AUC per fold stored in
emlm_folds.csv (see ensemble.py -- save_auc=True).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DS = sys.argv[1] if len(sys.argv) > 1 else "dataset2"
WHAT = sys.argv[2] if len(sys.argv) > 2 else "all"
RES = Path("results")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
MODELS = ["KNN", "SVM", "DT", "RF", "GNB", "AB"]
COLORS = {"I": "#1f77b4", "N": "#ff7f0e", "Z": "#2ca02c"}


def _fig5_axis(ax, df, group_col, groups, title):
    for g in groups:
        sub = df[df[group_col] == g].set_index("model").reindex(MODELS)
        ax.plot(MODELS, sub["aac"].values, marker="o", label=f"prep {g}", color=COLORS.get(g))
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("AAC")
    ax.grid(alpha=0.3)


def make_fig5():
    t5 = pd.read_csv(RES / f"{DS}_table5.csv")
    t6 = pd.read_csv(RES / f"{DS}_table6.csv")
    t7 = pd.read_csv(RES / f"{DS}_table7.csv")
    t8 = pd.read_csv(RES / f"{DS}_table8.csv")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    _fig5_axis(axes[0], t5, "prep", ["I", "N", "Z"], "(a) all features")
    # b/c/d: بهترین ترکیب فیچر-سلکشن برای هر (prep, model) — همان‌طور که مقاله بهترین را نشان می‌دهد
    for ax, df, col, ttl in [(axes[1], t6, "k", "(b) mRMR (best k)"),
                              (axes[2], t7, "var", "(c) PCA (best var)"),
                              (axes[3], t8, "k", "(d) ICA (best k)")]:
        best = df.loc[df.groupby(["prep", "model"])["aac"].idxmax()]
        _fig5_axis(ax, best, "prep", sorted(best["prep"].unique()), ttl)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{DS}: AAC across models — Fig.5 reproduction (4 experiments)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / f"{DS}_fig5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"{DS}_fig5.png"


def make_fig6a():
    mem = pd.read_csv(RES / f"{DS}_emlm_members.csv")
    t9 = pd.read_csv(RES / f"{DS}_table9.csv")
    labels = [f"{r.model}\n{r.prep}/{r.selector[:8]}" for r in mem.itertuples()]
    aucs = list(mem["auc"]) + [float(t9["auc"].iloc[0])]
    labels = labels + ["EMLM\n(weighted)"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, aucs, color=["#8ab4f8"] * len(mem) + ["#f28b82"])
    ax.set_ylim(min(aucs) - 0.05, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title(f"{DS}: Fig.6a reproduction — member AUCs vs optimal EMLM")
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"{DS}_fig6a.png", dpi=150)
    plt.close(fig)
    return f"{DS}_fig6a.png"


def make_fig6b():
    """ROC منحنیِ EMLM از روی احتمالات OOF (emlm_oof.csv، ستون‌های P_*).
    میانگین منحنی روی فولدهای بیرونی با درون‌یابی روی شبکه FPR."""
    from sklearn.metrics import roc_curve
    oof = pd.read_csv(RES / f"{DS}_emlm_oof.csv")
    pcols = [c for c in oof.columns if c.startswith("P_")]
    if not pcols:
        return "SKIPPED — emlm_oof.csv has no P_* columns (needs ensemble.py prob-save + nb03 re-run)"
    classes = sorted(int(c[2:]) for c in pcols)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6, 5))
    fold_curves = []
    for fold, sub in oof.groupby("fold"):
        tprs = []
        for cl, pc in zip(classes, pcols):
            yb = (sub["y_true"].values == cl).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            fpr, tpr, _ = roc_curve(yb, sub[pc].values)
            tprs.append(np.interp(grid, fpr, tpr))
        if tprs:
            fold_curves.append(np.mean(tprs, axis=0))
    for tpr in fold_curves:
        ax.plot(grid, tpr, alpha=0.15, color="#c0392b", lw=0.8)
    mean = np.mean(fold_curves, axis=0)
    mean_auc = float(np.trapezoid(mean, grid))
    ax.plot(grid, mean, color="#c0392b", lw=2.2,
            label=f"EMLM mean ROC (AUC≈{mean_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    ax.set_xlabel("False Positive Rate (one-vs-rest mean)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{DS}: Fig.6b reproduction — ROC of optimal EMLM ({len(fold_curves)} outer folds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"{DS}_fig6b.png", dpi=150)
    plt.close(fig)
    return f"{DS}_fig6b.png"



def make_fig_compare3():
    """نمودار مقایسه‌ی سه‌گانه: EMLM مقاله روی IPDD انتشاریافته، بازتولید ما روی IPDD عمومی،
    بازتولید ما روی پایگاه چینی — سه معیار AAC/ACC/AUC."""
    import pandas as pd
    t92 = pd.read_csv(RES / "dataset2_table9.csv").iloc[0]
    t91 = pd.read_csv(RES / "dataset1_table9.csv").iloc[0]
    labels = ["AAC", "ACC", "AUC"]
    series = {
        "Paper EMLM (IPDD published)": [98.5, 99.87, 99.9],
        f"Repro EMLM (IPDD public)": [t92["aac"] * 100, t92["acc"] * 100, t92["auc"] * 100],
        "Repro EMLM (China ds1)": [t91["aac"] * 100, t91["acc"] * 100, t91["auc"] * 100],
    }
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#1f4e79", "#2e8b57", "#b8860b"]
    for i, (k, v) in enumerate(series.items()):
        b = ax.bar(x + (i - 1) * w, v, w, label=k, color=colors[i])
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(60, 102); ax.set_ylabel("percent")
    ax.set_title("Three-way comparison of the EMLM: paper vs our two datasets")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    p = FIG / "fig_compare3.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return {p.name: str(p)}


def make_fig7():
    t9 = pd.read_csv(RES / f"{DS}_table9.csv")
    ours = float(t9["acc"].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["Hassan et al.\nHybrid KNN+ID3 (paper)", "Abnoosian et al.\nEMLM (paper)", "This work\nEMLM (repro)"]
    vals = [98.25, 99.87, round(ours * 100, 2)]
    bars = ax.bar(names, vals, color=["#c0c0c0", "#8ab4f8", "#f28b82"])
    ax.set_ylim(95, 100.6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{DS}: Fig.7 reproduction — ACC vs related work (Table 10)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_acc_comparison.png", dpi=150)
    plt.close(fig)
    return "fig7_acc_comparison.png"


MADE = {"all": [make_fig5, make_fig6a, make_fig6b, make_fig7],
        "fig5": [make_fig5], "fig6": [make_fig6a, make_fig6b],
        "fig6b": [make_fig6b], "fig7": [make_fig7],
        "compare3": [make_fig_compare3]}[WHAT]
for fn in MADE:
    print("wrote:", fn())

