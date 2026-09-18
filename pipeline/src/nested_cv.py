"""Nested CV مطابق Fig.4 مقاله: outer StratifiedKFold برای تست، inner برای tuning.
FAST=True: 2×2×1 + گرید کوچک (دود) | FAST=False: 5×4×1 + بیزی n_iter=20 («سطح متوازن»).
سطحِ کامل مقاله (5×4×5، n_iter=30): evaluate_cell(..., repeats=5) صریح — چرا کاهش دادیم؟
see docs/DEVIATIONS.md. Search parallelism: PARALLEL_N cores."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from models import GRIDS_FULL, GRIDS_SMALL, SCORING, make_pipe

# تعداد هسته برای موازی‌سازی جستجوی هایپرپارامتر (از ۲۰ رشته‌ی CPU).
# پیش‌فرض 8: همپوشانی معقول با کارهای دیگر سیستم؛ با متغیر محیطی قابل تغییر.
PARALLEL_N = int(os.environ.get("EMJM_NJOBS", "8"))


def _make_search(fast, model_name, pipe, inner, seed, n_samples=None):
    n_jobs = 1 if fast else PARALLEL_N
    grid = GRIDS_FULL[model_name]
    if (not fast) and model_name == "SVM" and n_samples and n_samples > 2000:
        grid = GRIDS_FULL["SVM__LITE"]  # see docs/DEVIATIONS.md
    if fast:
        return GridSearchCV(pipe, GRIDS_SMALL[model_name], cv=inner, scoring=SCORING, n_jobs=1)
    try:
        from skopt import BayesSearchCV
        return BayesSearchCV(pipe, grid, n_iter=20, cv=inner,
                             scoring=SCORING, n_jobs=n_jobs, random_state=seed)
    except ImportError:
        return GridSearchCV(pipe, grid, cv=inner, scoring=SCORING, n_jobs=n_jobs)


def evaluate_cell(X, y, prep, selector, model_name, fast=True, seed=42,
                  n_outer=None, n_inner=None, repeats=None):
    """یک سلول جدول = (prep, selector, model). خروجی: (فولدها, میانگین‌ها, std کلاس)
    سطح متوازن (پیش‌فرض FAST=False): 5×4×1 — برای کاملِ مقاله repeats=5 بدهید."""
    X = X.astype(float)
    labels = sorted(np.unique(y))
    if fast:
        n_outer, n_inner, repeats = n_outer or 2, n_inner or 2, repeats or 1
    else:
        n_outer, n_inner, repeats = n_outer or 5, n_inner or 4, repeats or 1

    rows = []
    for r in range(repeats):
        outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed + r)
        for tr, te in outer.split(X, y):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
            inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + 100 + r)
            gs = _make_search(fast, model_name, make_pipe(prep, model_name, selector), inner, seed,
                          n_samples=Xtr.shape[0])
            t0 = time.time()
            gs.fit(Xtr, ytr)
            best = gs.best_estimator_
            proba = best.predict_proba(Xte)
            pred = best.predict(Xte)
            if len(labels) >= 3:
                auc = roc_auc_score(yte, proba, multi_class="ovo", average="macro", labels=labels)
            else:
                auc = roc_auc_score(yte, proba[:, 1])
            rows.append(dict(repeat=r,
                             aac=balanced_accuracy_score(yte, pred),      # Eq.8 مقاله
                             acc=accuracy_score(yte, pred),
                             precision=precision_score(yte, pred, average="micro", zero_division=0),
                             recall=recall_score(yte, pred, average="micro", zero_division=0),
                             f1=f1_score(yte, pred, average="micro", zero_division=0),
                             auc=auc, fit_s=round(time.time() - t0, 2),
                             best_params=json.dumps(gs.best_params_)))
    df = pd.DataFrame(rows)
    agg = {m: round(float(df[m].mean()), 4) for m in ["aac", "acc", "precision", "recall", "f1", "auc"]}
    return df, agg, float(df["aac"].std(ddof=0))


def append_checkpoint(path, dataset, exp, prep, selector_name, model,
                      aac_mean, aac_std, auc_mean, n_rows, extra=""):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([dict(dataset=dataset, exp=exp, prep=prep, selector=selector_name,
                             model=model, aac_mean=round(float(aac_mean), 4),
                             aac_std=round(float(aac_std), 4), auc_mean=round(float(auc_mean), 4),
                             n_rows=n_rows, extra=extra,
                             ts=pd.Timestamp.now().isoformat(timespec="seconds"))])
    row.to_csv(p, mode="a", header=not p.exists(), index=False, encoding="utf-8-sig")
