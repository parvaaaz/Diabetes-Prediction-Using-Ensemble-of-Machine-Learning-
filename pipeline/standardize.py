"""
standardize.py — قالب استاندارد مشترک برای کار روی ۳ دیتاست دیابت (پروژه بازتولید مقاله BMC Bioinformatics 2023)

هدف:
    هر دیتاست (Dataset1 چینی، Dataset2 عراقی/IPDD، Dataset3 آینده/IEEE DataPort) را به یک
    «اسکیمای کنونیکال» یکسان تبدیل می‌کنیم تا همه‌ی نوت‌بوک‌ها و مدل‌ها بدون تغییر کد
    روی هر سه اجرا شوند.

اسکیمای کنونیکال (ترتیب ستون‌ها ثابت و در همه دیتاست‌ها یکسان):
    ---------------------------------------------------------------
    col 1  : ID            شناسه ردیف (int، از فایل اصلی)
    col 2  : Gender        کد استاندارد: 0 = زن (Female) ، 1 = مرد (Male)
    col 3  : AGE           سن (سال کامل)
    col 4  : Urea          اوره خون (mmol/L)
    col 5  : Cr            کراتینین سرم (µmol/L)
    col 6  : FBS           قند خون ناشتا (mmol/L)
    col 7  : HbA1c         هموگلوبین A1c (٪)
    col 8  : Chol          کلسترول توتال (mmol/L)
    col 9  : TG            تری‌گلیسرید (mmol/L)
    col 10 : HDL           کلسترول خوب (mmol/L)
    col 11 : LDL           کلسترول بد (mmol/L)
    col 12 : VLDL          کلسترول VLDL (mmol/L) — اگر دیتاستی نداشت NaN
    col 13 : BMI           شاخص توده بدنی (kg/m²)
    col 14 : CLASS_code    کد برچسب سه‌کلاسه استاندارد: 0 = N (سالم) ، 1 = P (پردیابت) ، 2 = Y (دیابتی)
    col 15 : CLASS         برچسب متنی: 'N' / 'P' / 'Y'
    ---------------------------------------------------------------

نحوه‌ی استفاده:
    1) برای هر دیتاست یک فایل کانفیگ JSON در پوشه‌ی configs/ بساز (نمونه: configs/dataset1.json)
    2) اجرا:  python standardize.py configs/dataset1.json
    3) خروجی: data/<name>_standard.csv  +  گزارش راستی‌آزمایی در خروجی ترمینال

هر دیتاست ممکن است فیچری را نداشته باشد (مثلاً VLDL در Dataset1 وجود ندارد): ستون با NaN ساخته
می‌شود و در گزارش اعلام می‌شود تا در imputation (kNN طبق مقاله) پر شود.
همچنین اگر دیتاستی FBS نداشت ولی VLDL داشت (مثل Dataset2) مشکلی نیست؛ اسکیما هر دو را دارد.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# اسکیمای کنونیکال — ترتیب و نام ستون‌های خروجی (ثابت برای همه دیتاست‌ها)
# ---------------------------------------------------------------------------
CANONICAL_COLS = [
    "ID", "Gender", "AGE", "Urea", "Cr", "FBS", "HbA1c",
    "Chol", "TG", "HDL", "LDL", "VLDL", "BMI",
    "CLASS_code", "CLASS",
]

GENDER_MAP_OPTIONS = {
    # نگاشت‌های رایج: هر کدام در کانفیگ انتخاب می‌شود
    "M_F":   {"M": 1, "F": 0},
    "1_2":   {1: 1, 2: 0},          # Dataset1: 1=Male، 2=Female
    "0_1":   {0: 0, 1: 1},          # 0=Female، 1=Male
    "1_0":   {1: 0, 0: 1},          # 1=Female، 0=Male
}

CLASS_MAP_OPTIONS = {
    "N_P_Y": {"N": 0, "P": 1, "Y": 2},        # Dataset2
    "0_1_2": {0: 0, 1: 1, 2: 2},              # Dataset1 (PGclass3)
    "0_1":   {0: 0, 1: 2},                    # دودویی: 0=سالم، 1=دیابتی (پردیابت ندارد)
}


def _to_num(s: pd.Series) -> pd.Series:
    """تبدیل امن به عددی (مقادیر خالی/متنی به NaN)."""
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")


def standardize(cfg_path: str) -> pd.DataFrame:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    name = cfg["name"]
    src = Path(cfg["path"])
    sep = cfg.get("sep", ",")
    enc = cfg.get("encoding", "utf-8")
    df = pd.read_csv(src, sep=sep, encoding=enc, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    colmap = cfg["columns"]           # نگاشت: نام کنونیکال -> نام ستون در فایل خام (یا null اگر ندارد)
    gmap = GENDER_MAP_OPTIONS[cfg["gender_mapping"]]
    cmap = CLASS_MAP_OPTIONS[cfg["class_mapping"]]

    out = pd.DataFrame(index=df.index)

    # --- ID ---
    id_col = colmap.get("ID")
    out["ID"] = df[id_col] if id_col else np.arange(1, len(df) + 1)

    # --- Gender: نرمال‌سازی به 0=زن، 1=مرد ---
    g = df[colmap["Gender"]]
    if gmap is GENDER_MAP_OPTIONS["M_F"]:
        g = g.astype(str).str.strip().str.upper().replace({"F": "F", "M": "M"})
    g_std = g.map(gmap)
    if g_std.isna().any():
        bad = g[g_std.isna()].unique()
        print(f"  [warn] {name}: {len(g_std) - g_std.notna().sum()} ردیف جنسیت ناشناخته {bad} -> NaN")
    out["Gender"] = g_std.astype("Int64")

    # --- فیچرهای عددی ---
    for c in ["AGE", "Urea", "Cr", "FBS", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]:
        src_col = colmap.get(c)
        if src_col is None or src_col not in df.columns:
            out[c] = np.nan
            print(f"  [info] {name}: ستون '{c}' در دیتاست اصلی وجود ندارد -> NaN (بعداً با kNN impute پر می‌شود)")
        else:
            out[c] = _to_num(df[src_col])

    # --- برچسب سه‌کلاسه ---
    y = df[colmap["CLASS"]]
    if cmap is CLASS_MAP_OPTIONS["N_P_Y"]:
        y = y.astype(str).str.strip().str.upper()
    y_std = y.map(cmap)
    if y_std.isna().any():
        bad = y[y_std.isna()].unique()
        raise ValueError(f"{name}: برچسب ناشناخته {bad} — class_mapping را اصلاح کن")
    out["CLASS_code"] = y_std.astype(int)
    out["CLASS"] = out["CLASS_code"].map({0: "N", 1: "P", 2: "Y"})

    out = out[CANONICAL_COLS]

    # --- خروجی ---
    dst = Path(cfg.get("output", f"data/{name}_standard.csv"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8-sig")

    # --- گزارش راستی‌آزمایی ---
    print(f"\n=== {name} -> {dst} ===")
    print(f"shape: {out.shape}")
    print("CLASS dist:", out["CLASS"].value_counts().reindex(["N", "P", "Y"]).fillna(0).astype(int).to_dict())
    print("Gender dist (0=زن, 1=مرد):", out["Gender"].value_counts().sort_index().to_dict())
    miss = out.isna().sum()
    miss = miss[miss > 0]
    print("missing:", miss.to_dict() if len(miss) else "ندارد")
    print("means (مقایسه با Table 1 مقاله برای IPDD):")
    print(out[["AGE", "Urea", "Cr", "FBS", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]]
          .mean().round(2).to_string())
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python standardize.py configs/<dataset>.json")
        sys.exit(1)
    standardize(sys.argv[1])
