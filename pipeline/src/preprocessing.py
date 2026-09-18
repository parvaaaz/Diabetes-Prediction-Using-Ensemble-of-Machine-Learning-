"""Preprocessing طبق مقاله (Section 2.1-2.3): dedup, kNN imputation, normalization/standardization, mRMR."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif

FEATURES_ALL = ["Gender", "AGE", "Urea", "Cr", "FBS", "HbA1c", "Chol",
                "TG", "HDL", "LDL", "VLDL", "BMI"]


def load_standard(csv_path):
    """بارگذاری CSV استاندارد؛ فیچرهای 100%-خالی (مثل FBS در dataset2 / VLDL در dataset1) کنار گذاشته می‌شوند."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    feats = [c for c in FEATURES_ALL if df[c].notna().any()]
    return df, feats


def dedup(df, strategy="features"):
    """استراتژی primary پلن: حذف تکراری روی ۱۱ فیچر موجود + کلاس.
    dataset2: 1000 -> 826 | dataset1: بدون تغییر (تکراری ندارد)."""
    cols = [c for c in FEATURES_ALL if c in df.columns and df[c].notna().any()] + ["CLASS_code"]
    n0 = len(df)
    out = df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    print(f"dedup[{strategy}]: {n0} -> {len(out)}")
    return out


class MRMRSelector(BaseEstimator, TransformerMixin):
    """mRMR تقریبی (بدون وابستگی خارجی): relevance = F-score آنوا، redundancy = میانگین |همبستگی| با انتخاب‌شده‌ها.
    نمره‌ی greedy:  F_j * (1 - mean|corr(j, selected)|)   — تقریب MID؛ در گزارش مستند می‌شود."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            F, _ = f_classif(X, y)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0)
        C = np.abs(np.corrcoef(X, rowvar=False))
        C = np.nan_to_num(C, nan=0.0)
        n = X.shape[1]
        k = int(min(self.k, n))
        selected = [int(np.argmax(F))]
        while len(selected) < k:
            best_j, best_s = None, -np.inf
            for j in range(n):
                if j in selected:
                    continue
                red = float(C[j, selected].mean())
                s = float(F[j]) * (1.0 - red)
                if s > best_s:
                    best_j, best_s = j, s
            selected.append(int(best_j))
        self.selected_ = np.array(selected, dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.selected_]
