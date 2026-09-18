"""EMLM — کمیته‌ی وزن‌دار مدل‌ها بر اساس AUC (بازتولید جداول ۹ و ۱۰ مقاله).

وزن‌دهی: مقاله وزن هر مدل را AUC_i / ΣAUC می‌گیرد. ما برای جلوگیری از نشت اطلاعات،
AUCِ هر عضو را از میانگین AUC درونیِ همان tuning (inner-CV) می‌گیریم و کف 0.5
(شانس تصادفی) می‌گذاریم تا عضوِ بد وزن منفی/بی‌اثر نگیرد. در md مستند شده.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from models import make_pipe, selector_from_name
from nested_cv import _make_search


def build_committee(results_dir, dataset):
    """اعضای کمیته = بهترین سلول هر یک از ۴ آزمایش (جداول ۵-۸ نوت‌بوک ۲)."""
    t5 = pd.read_csv(f"{results_dir}/{dataset}_table5.csv")
    t5["selector"] = "all_features"
    t6 = pd.read_csv(f"{results_dir}/{dataset}_table6.csv")
    t6["selector"] = "mrmr_" + t6["k"].astype(int).astype(str)
    t7 = pd.read_csv(f"{results_dir}/{dataset}_table7.csv")
    t7["selector"] = "pca_" + t7["var"].astype(str)
    t8 = pd.read_csv(f"{results_dir}/{dataset}_table8.csv")
    t8["selector"] = "ica_" + t8["k"].astype(int).astype(str)

    rows, seen = [], set()
    for exp, df in [("exp1_table5", t5), ("exp2_table6", t6),
                    ("exp3_table7", t7), ("exp4_table8", t8)]:
        b = df.loc[df["aac"].idxmax()]
        key = (b["prep"], b["selector"], b["model"])
        if key in seen:      # two experiments can land on the same config
            continue
        seen.add(key)
        rows.append(dict(exp=exp, prep=b["prep"], selector=b["selector"],
                         model=b["model"], aac_nb02=b["aac"]))
    return pd.DataFrame(rows)


def evaluate_committee(X, y, members, fast=True, seed=42):
    """ارزیابی اعضا + EMLM روی فولدهای بیرونیِ مشترک (تا ترکیبِ احتمالات معنادار باشد).
    خروجی: (جدول اعضا, خلاصه‌ی EMLM, ردیف‌به‌ردیف EMLM, پیش‌بینی‌های OOF)"""
    labels = sorted(np.unique(y))
    if fast:
        n_outer, n_inner, repeats = 2, 2, 1
    else:
        # Budget: 5x4x1 with n_iter=20 (5x4x5 is impractical for dataset1,
        # where SVM is slow at n=5922). See docs/DEVIATIONS.md.
        n_outer, n_inner, repeats = 5, 4, 1
    X = X.astype(float).reset_index(drop=True)
    y = y.reset_index(drop=True)

    mem_rows, emlm_rows, oof = [], [], []
    for r in range(repeats):
        outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed + r)
        for tr, te in outer.split(X, y):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + 100 + r)

            weighted, wsum, classes_ref = [], 0.0, None
            for _, m in members.iterrows():
                sel = selector_from_name(m["selector"])
                gs = _make_search(fast, m["model"],
                                  make_pipe(m["prep"], m["model"], sel), inner, seed,
                                  n_samples=Xtr.shape[0])
                gs.fit(Xtr, ytr)
                best = gs.best_estimator_
                if classes_ref is None:
                    classes_ref = best.classes_
                w = max(float(gs.cv_results_["mean_test_score"][gs.best_index_]), 0.5)
                P = best.predict_proba(Xte)
                pred = best.predict(Xte)
                auc = (roc_auc_score(yte, P, multi_class="ovo", average="macro", labels=labels)
                       if len(labels) >= 3 else roc_auc_score(yte, P[:, 1]))
                mem_rows.append(dict(exp=m["exp"], prep=m["prep"], selector=m["selector"],
                                     model=m["model"],
                                     aac=balanced_accuracy_score(yte, pred),
                                     acc=accuracy_score(yte, pred), auc=auc, weight=w))
                weighted.append(w * P)
                wsum += w

            P_emlm = sum(weighted) / wsum
            pred_emlm = classes_ref[np.argmax(P_emlm, axis=1)]
            auc_e = (roc_auc_score(yte, P_emlm, multi_class="ovo", average="macro", labels=labels)
                     if len(labels) >= 3 else roc_auc_score(yte, P_emlm[:, 1]))
            emlm_rows.append(dict(repeat=r,
                                  aac=balanced_accuracy_score(yte, pred_emlm),
                                  acc=accuracy_score(yte, pred_emlm), auc=auc_e))
            # OOF با احتمالات کلاس — ستون‌های P_* برای رسم ROC (Fig.6b مقاله)
            oof_row = {"y_true": yte.values, "y_pred": pred_emlm,
                       "fold": len(emlm_rows) - 1}
            for j, cl in enumerate(classes_ref):
                oof_row[f"P_{int(cl)}"] = P_emlm[:, j]
            oof.append(pd.DataFrame(oof_row))

    mem_df = (pd.DataFrame(mem_rows)
              .groupby(["exp", "prep", "selector", "model"], as_index=False)
              .agg(aac=("aac", "mean"), acc=("acc", "mean"),
                   auc=("auc", "mean"), weight=("weight", "mean"))
              .round(4))
    emlm_df = pd.DataFrame(emlm_rows)
    emlm_summary = dict(aac=round(float(emlm_df["aac"].mean()), 4),
                        acc=round(float(emlm_df["acc"].mean()), 4),
                        auc=round(float(emlm_df["auc"].mean()), 4),
                        aac_std=round(float(emlm_df["aac"].std(ddof=0)), 4),
                        n_folds=len(emlm_df))
    return mem_df, emlm_summary, emlm_df, pd.concat(oof, ignore_index=True)
