# Results

All numbers below are produced by the code in this repo and stored as CSV in
`pipeline/results/` — nothing is hand-typed. Reference: Abnoosian et al., BMC
Bioinformatics 2023 (paper protocol reproduced at the balanced 5×4×1 level,
see [docs/DEVIATIONS.md](DEVIATIONS.md)).

## Headline (paper Table 9 equivalent)

| dataset | protocol | AAC | ACC | AUC |
|---|---|---|---|---|
| **dataset2** (IPDD, n=826) | EMLM committee | **0.9366 ± 0.0379** | 0.9770 | 0.9970 |
| **dataset1** (Chinese, n=5922, generality check) | EMLM committee | **0.7362 ± 0.0124** | 0.8948 | 0.8660 |
| paper (private post-merge IPDD, n=993) | EMLM, 5×4×5 | 0.9850 | 0.9987 | 0.9990 |

The gap on dataset2 is fully discussed in `docs/DEVIATIONS.md` §7–8 — the main
drivers are the missing FBS column in the public IPDD file, the leakage-free
weighting, and the nested-CV protocol (the paper uses a single 80/20 hold-out).

## Committee (dataset2)

| experiment | prep | selector | model | AAC | AUC | weight |
|---|---|---|---|---|---|---|
| exp1 | I | all_features | DT | 0.9513 | 0.9916 | 0.9933 |
| exp2 | N | mrmr_4 | RF | 0.9637 | 0.9980 | 0.9975 |
| exp3 | N | pca_0.95 | GNB | 0.7311 | 0.9089 | 0.8934 |
| exp4 | N | ica_5 | SVM | 0.7558 | 0.9347 | 0.9377 |

Committee (dataset1): Z/RF ×2 (mrmr_10, all), Z/SVM pca_0.98, Z/RF ica_5 —
see `pipeline/results/dataset1_emlm_members.csv`.

## Dedup sensitivity (dataset2, same committee, same protocol)

| variant | n | AAC | AAC std | ACC | AUC |
|---|---|---|---|---|---|
| after dedup | 826 | 0.9366 | ±0.0379 | 0.9770 | 0.9970 |
| pre-dedup | 1000 | 0.9229 | ±0.0703 | 0.9720 | 0.9930 |

Strict dedup raises the mean and halves the fold-to-fold variance.

## External comparison (paper Table 10 line, plus our generality run)

| model | dataset | ACC % |
|---|---|---|
| Hybrid KNN+ID3 (Soukaena Hassan et al.) | Pima Indian (binary) | 98.25 |
| EMLM (paper) | IPDD private (multiclass) | 99.87 |
| **EMLM (this repro)** | IPDD public, 826 rows | **97.70** |
| **EMLM (this repro)** | Chinese, 5922 rows | **89.48** |

## Figures

- `figures/dataset{1,2}_fig5.png` — 4-panel AAC (all / mRMR / PCA / ICA), paper Fig. 5
- `figures/dataset{1,2}_fig6a.png` — member vs. EMLM AUC, paper Fig. 6a
- `figures/dataset{1,2}_fig6b.png` — mean-of-folds ROC from OOF probabilities, paper Fig. 6b
- `figures/fig7_acc_comparison.png` — the three-way ACC bar, paper Fig. 7
- `figures/pipeline_figure.png` — the actual executed pipeline (this project's diagram)
