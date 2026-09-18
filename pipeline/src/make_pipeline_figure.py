# -*- coding: utf-8 -*-
"""Draw the executed-pipeline figure (Dataset 1 & Dataset 2 only).

Run from pipeline/:  python src/make_pipeline_figure.py
All box contents mirror the actual code/results — every claim here has a
code citation:
  DATA rows      : pipeline/data/*.csv (audited counts), preprocessing.py
  PREPROCESSING  : preprocessing.load_standard/dedup (keep="first")
  FEATURES       : models.make_pipe (KNNImputer(5) -> I/N/Z scaler -> mRMR/PCA/ICA)
  NESTED band    : nested_cv.evaluate_cell (StratifiedKFold 5x4x1, seed 42,
                   BayesSearchCV n_iter=20, scoring=roc_auc_ovo, SVM-lite >2000 rows)
  CLASSIFIERS    : models.MODELS (KNN/SVM/DT/RF/GNB/AB)
  EMLM           : ensemble.py (4 experts, weight = member OVO-AUC, same nested eval)
  NUMBERS        : results/dataset{1,2}_table9.csv (printed into the figure)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

RES = Path(__file__).resolve().parent.parent / "results"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

n2 = pd.read_csv(RES / "dataset2_table9.csv").iloc[0]
n1 = pd.read_csv(RES / "dataset1_table9.csv").iloc[0]


def fa(x):  # straight western digits — readable at print size
    return f"{x:.4f}"


BAND_PAIRS = set()  # band is drawn standalone, never registered in boxes

# ---------------------------------------------------------------- canvas
fig, ax = plt.subplots(figsize=(6.6, 4.1), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

C_DATA, C_PRE, C_FEAT = "#eaf2fb", "#eef7ee", "#fdf3e7"
C_BAND, C_MODEL, C_EMLM, C_RES = "#f4f0fa", "#e7f0fa", "#fdecec", "#e8f5f0"
EDGE = "#4a4a4a"

boxes, texts = [], []


def box(x, y, w, h, fc, label_lines, fs=6.2, ec=EDGE, dashed=False, bold_first=True,
        title=None):
    ls = "--" if dashed else "-"
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.8, ls=ls,
                           zorder=2))
    boxes.append((x, y, w, h))
    n = len(label_lines)
    for i, ln in enumerate(label_lines):
        bold = bold_first and i == 0
        t = ax.text(x + w / 2, y + h - (i + 0.62) * h / (n + 0.25), ln, ha="center",
                    va="center", fontsize=(fs + 0.5) if bold else fs,
                    fontweight="bold" if bold else "normal", zorder=3)
        texts.append(t)
    if title:
        ax.text(x + w / 2, y + h + 1.4, title, ha="center", va="bottom", fontsize=7,
                fontweight="bold", color="#333")
    return (x, y, w, h)


def arrow(p1, p2, dashed=False, color="#555"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, linestyle="--" if dashed else "-",
                                 color=color, zorder=1,
                                 shrinkA=1, shrinkB=1))
    return (p1, p2)


# ---------------------------------------------------------------- DATA row
b_ds1 = box(2, 84.5, 41, 13.5, C_DATA, [
    "DATASET 1 — Chinese cohort",
    "(Diabetes Risk Data, Kaggle)",
    "n = 5,922 (N 4,444 / P 1,121 / Y 357)",
    "VLDL 100% missing → 11 usable features",
])
b_ds2 = box(57, 84.5, 41, 13.5, C_DATA, [
    "DATASET 2 — IPDD public (Mendeley)",
    "raw n = 1,000 (N 103 / P 53 / Y 844)",
    "FBS 100% missing → 11 usable features",
])
box(44.0, 85.5, 12, 11.5, "#f2f2f2", ["ref only:", "private IPDD", "(paper, n=993)"],
    fs=5.0, ec="#aaa")

# ---------------------------------------------------------------- PREPROCESSING
b_pre = box(16, 69.0, 68, 13.5, C_PRE, [
    "PREPROCESSING (standardize.py + load time)",
    "drop 100%-empty cols · dedup exact rows (11 feats + class, keep first)",
    "DS2: 1,000 → 826 (174 dups; sensitivity-tested) · DS1: unchanged",
    "labels N/P/Y → 0/1/2",
])
arrow((22, 84.5), (38, 82.5))          # DS1 -> pre
arrow((77, 84.5), (62, 82.5))          # DS2 -> pre

# ---------------------------------------------------------------- FEATURES
b_feat = box(16, 55.5, 68, 12.0, C_FEAT, [
    "FEATURE LAYER — fit only inside inner-CV folds",
    "impute: KNNImputer(k=5) · scale: I raw / N MinMax / Z Standard",
    "reduce: all · mRMR(k∈{4,6,8,10}) · PCA(v∈{.90,.95,.98}) · ICA(k∈{4,5})",
])
arrow((50, 69.5), (50, 67.5))

# ---------------------------------------------------------------- NESTED CV band
ax.add_patch(Rectangle((6, 27.5), 88, 26.5, facecolor=C_BAND, edgecolor=EDGE,
                       lw=1.0, ls="--", zorder=0))
ax.text(50, 51.9, "NESTED CV — outer 5-fold × inner 4-fold × 1 repeat (paper: 5×4×5)",
        ha="center", fontsize=6.4, fontweight="bold")
b_tune = box(9, 37.0, 37, 11.0, "#ffffff", [
    "Hyper-param search (inner folds)",
    "BayesSearchCV n_iter=20",
    "scoring = roc_auc_ovo · SVM grid",
    "C/γ only when n>2,000 (DS1)",
], fs=5.6)
b_models = box(50, 37.0, 41, 11.0, C_MODEL, [
    "6 classifiers (sklearn)",
    "KNN · SVM · DT · RF · GNB · AdaBoost",
    "each × I/N/Z × every reduce option",
], fs=5.6)
b_metrics = box(9, 28.0, 82, 7.5, "#ffffff", [
    "per outer fold (test part unseen): AAC (Eq.8) · ACC · AUC (OVO macro)",
    "tuning never touches the outer test fold — no leakage path",
], fs=5.8, bold_first=False)
arrow((47, 42.5), (50, 42.5))
arrow((28, 37.0), (28, 35.3))
arrow((70, 37.0), (70, 35.3))
arrow((50, 55.5), (50, 54.0))

# ---------------------------------------------------------------- EMLM
b_emlm = box(16, 15.0, 68, 10.0, C_EMLM, [
    "EMLM — weighted ensemble (paper Sec. 2.5)",
    "4 experts (best per Exp.1–4) · weight = member's inner-CV",
    "OVO-AUC (floor 0.5, no test leakage) · weighted vote",
], fs=5.8)
arrow((50, 27.5), (50, 25.0))

# ---------------------------------------------------------------- RESULTS
b_r2 = box(2, 1.0, 40, 11.5, C_RES, [
    f"RESULT DS2 (n=826)",
    f"AAC {fa(n2.aac)}±{fa(n2.aac_std)}",
    f"ACC {fa(n2.acc)}   AUC {fa(n2.auc)}",
], fs=6.0)
b_r1 = box(50, 1.0, 48, 11.5, C_RES, [
    f"RESULT DS1 (n=5,922)",
    f"AAC {fa(n1.aac)}±{fa(n1.aac_std)}",
    f"ACC {fa(n1.acc)}   AUC {fa(n1.auc)}",
], fs=6.0)
arrow((30, 15.0), (22, 12.5))
arrow((70, 15.0), (74, 12.5))
ax.text(50, 0.2, "external benchmark (not reproduced): paper EMLM on private IPDD — "
                 "AAC 0.9850 / ACC 0.9987 / AUC 0.9990", ha="center", fontsize=5.4,
        color="#666", style="italic")

# ---------------------------------------------------------------- geometry audit
fig.canvas.draw()
rend = [t.get_window_extent(renderer=fig.canvas.get_renderer()) for t in texts]
problems = []
for i in range(len(rend)):
    for j in range(i + 1, len(rend)):
        if rend[i].overlaps(rend[j]):
            problems.append(f"text overlap {i}x{j}")
# boxes must not overlap each other
bp = [ax.transData.transform([(x, y), (x + w, y + h)]) for x, y, w, h in boxes]
for i in range(len(bp)):
    for j in range(i + 1, len(bp)):
        (ax0, ay0), (ax1, ay1) = bp[i]
        (bx0, by0), (bx1, by1) = bp[j]
        r1 = matplotlib.transforms.Bbox.from_extents(min(ax0, ax1), min(ay0, ay1), max(ax0, ax1), max(ay0, ay1))
        r2 = matplotlib.transforms.Bbox.from_extents(min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))
        # band is intentionally a container; skip band-vs-box checks via sentinel list
        if r1.overlaps(r2) and (i, j) not in BAND_PAIRS:
            problems.append(f"box overlap {i}x{j}")
print("AUDIT boxes:", len(boxes), "texts:", len(texts), "overlaps:", len(problems))
out = FIG / "pipeline_figure.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
print("SAVED:", out, out.stat().st_size, "bytes")
if problems:
    print("PROBLEMS:", *problems, sep="\n  ")
    raise SystemExit(1)
