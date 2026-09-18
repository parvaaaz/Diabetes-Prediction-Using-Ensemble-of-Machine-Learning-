# Deviations from the reference paper

This repository reproduces the methodology of:

> Abnoosian K., Farnoosh R., Behzadi M.H. *Prediction of diabetes disease using an
> ensemble of machine learning multi-classifier models (EMLM).* BMC Bioinformatics
> 24:337 (2023). <https://doi.org/10.1186/s12859-023-05465-z>

Every deliberate departure from the published protocol is recorded here with its
reason. Governing rule of the project: **no change without a documented rationale.**

---

## 1. Nested-CV budget: 5×4×5 → 5×4×1 ("balanced level")

**Where:** `pipeline/src/nested_cv.py` (`evaluate_cell`, `FAST=False`)

The paper uses 5 outer folds × 4 inner folds × 5 repeats. We run 5 × 4 × **1**
and reduce the Bayesian search budget from 30 to **20** iterations.

The three knobs are not equally expensive:

- *repeats* only shrink the variance of the estimate; the mean AAC barely moves —
  the cheapest thing to give up;
- *n_iter*: on these dataset sizes (826 / 5922 rows) Bayesian optimization
  converges around iteration ~15;
- *outer folds*: the backbone of the error estimate — **untouched**, so comparison
  with the paper's numbers stays meaningful.

Expected effect: means in the same family as the paper's, fold-to-fold std slightly
coarser. Full paper level is one argument away:

```python
evaluate_cell(X, y, prep, sel, model, fast=False, repeats=5)
```

## 2. Parallelism of the hyper-parameter search

`PARALLEL_N` cores (default 8) run the Bayesian search; override with the
`EMJM_NJOBS` environment variable. This is an implementation detail, not
methodology — no scientific deviation.

## 3. Two inert axes removed from the KNN grid

**Where:** `pipeline/src/models.py` → `GRIDS_FULL["KNN"]`

Removed: `algorithm` (auto/ball_tree/kd_tree/brute) and `leaf_size`. These only
change the neighbor-search data structure; predictions are bit-identical. Without
the removal, ~1600 extra no-op configurations would dilute the Bayesian budget.
All other grids (SVM/DT/RF/GNB/AB) match the paper verbatim.

## 4. EMLM weighting: leakage-free, floored

**Where:** `pipeline/src/ensemble.py`

The paper weights each committee member by its test AUC (`AUC_i / Σ AUC`). We
instead use `w_i = max(AUC_i^inner, 0.5)`:

1. the AUC comes from the **inner CV** of the tuning loop, never from the outer
   test fold (the paper's scheme leaks test information into the weights);
2. a floor of 0.5 stops an at-chance member from getting a destructive weight.

Committee members are re-weighted **per outer fold**; nothing is fitted on the
outer test data.

## 5. OOF class probabilities are stored

`results/<ds>_emlm_oof.csv` carries `P_0/P_1/P_2` so the paper's mean-OOF ROC
(Fig. 6b) can be redrawn and independently verified (AUC recomputed from stored
probabilities with `sklearn.roc_curve`; the pairwise `roc_auc_score` values match a
standalone Mann–Whitney U reconstruction to machine precision).

## 6. Restricted SVM grid for the large dataset (n > 2000)

**Where:** `GRIDS_FULL["SVM__LITE"]`, applied in `nested_cv.py` only when `n > 2000`

Direct benchmarking on the 4737-row train slice of dataset1 showed some
paper-grid SVM combinations (`poly`, `sigmoid`, `C=100/gamma=1`) never finished
within 590 s **per single fit**. Two benign fixes:

- `SVC(cache_size=1024)` — kernel-matrix cache only; deterministic predictions;
- `SVM__LITE`: RBF kernel, `C ∈ [0.1, 10]`, `gamma ∈ [0.01, 0.1]` (log-uniform).
  Linear kernel was dropped after measurement (C≈9.9 linear ≈ 96 s/fit; worst RBF
  fit < 4 s).

dataset2 (the dataset the paper's numbers refer to) keeps the **full, untouched
paper grid** — the headline comparison is unaffected. This adaptation only touches
dataset1, used as our generality check.

Related finding: nested parallelism (8 Bayesian workers × 20 BLAS threads) caused
CPU oversubscription — RandomForest runs with `n_jobs=1` and BLAS thread pinning.

## 7. Deduplication sensitivity — a self-claim the data disproved

**Script:** `run_dedup_sensitivity.py` → `results/dedup_sensitivity.csv`

We initially claimed the 174 rows removed by our strict dedup (vs. the paper's 7)
explained the AAC gap between the paper's 0.985 and our 0.9366. The controlled run
(same committee, same 5×4×1 protocol, only the data variant differs) shows:

| variant | n | AAC | AAC std | ACC | AUC |
|---|---|---|---|---|---|
| after dedup (headline) | 826 | 0.9366 | ±0.0379 | 0.9770 | 0.9970 |
| pre-dedup | 1000 | 0.9229 | ±0.0703 | 0.9720 | 0.9930 |

**The hypothesis was rejected**: dedup *raises* the mean and roughly halves the
fold-to-fold variance. The claim was retracted from the write-up; remaining gap
attributions: leakage-free weighting, the missing FBS column in the public IPDD
version, and nested-CV vs. a plain train/test split. Lesson recorded: every
"cause of the gap" claim must be tested with a cheap counter-run before print.

## 8. Not reproduced (by design)

- **Dataset3 / third-dataset experiments:** not part of this reproduction.
- **LASSO / RFE / SVRFE:** discussed conceptually in the write-up as alternatives
  to mRMR, but **never executed**. Only mRMR, PCA and ICA run here.
- **The paper's private post-merge dataset** (993 rows, with the correspondence-only
  FBS column) is not available; dataset2 is the public IPDD version (no FBS).
