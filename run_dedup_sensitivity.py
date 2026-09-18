# -*- coding: utf-8 -*-
"""حساسیت‌سنجی dedup (مستندسازی ادعای فصل ۴):
EMLM روی dataset2 پیش از حذف تکراری‌ها (1000 رکورد) در برابر پس از dedup (826).
خروجی: pipeline/results/dedup_sensitivity.csv
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT / "pipeline")
sys.path.insert(0, str(ROOT / "pipeline" / "src"))

import pandas as pd
from preprocessing import load_standard, dedup
from ensemble import build_committee, evaluate_committee

rows = []
for tag, use_dedup in [("after_dedup", True), ("pre_dedup", False)]:
    df, FEATS = load_standard("data/dataset2_standard.csv")
    if use_dedup:
        df = dedup(df)
    X, y = df[FEATS], df["CLASS_code"]
    committee = build_committee("results", "dataset2")
    mem, summ, folds, oof = evaluate_committee(X, y, committee, fast=False)
    rows.append(dict(variant=tag, n=len(df), **summ))
    print(f"=== {tag}: n={len(df)} -> {summ}", flush=True)

out = pd.DataFrame(rows)
out.to_csv("results/dedup_sensitivity.csv", index=False)
print(out.to_string(index=False))
print("saved: results/dedup_sensitivity.csv")
