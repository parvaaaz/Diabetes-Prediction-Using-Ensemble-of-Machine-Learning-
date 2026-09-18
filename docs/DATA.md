# Datasets & provenance

Two public diabetes datasets, mapped to one canonical 15-column schema by
`pipeline/standardize.py` (configs in `pipeline/configs/`).

| | Dataset1 | Dataset2 |
|---|---|---|
| Cohort | Chinese hospital cohort, 5922 adults (18–75) | IPDD — Iraqi Patient Dataset for Diabetes, 1000 samples (20–79), Yersal et al. |
| Rows used (post-dedup) | 5922 (0 duplicates) | **826** (174 exact duplicates removed) |
| Classes (N/P/Y) | 4444 / 1121 / 357 | 103 / 53 / 837 (paper Table 1 counts) |
| 11 features | Gender, AGE, Urea, Cr, **FBS**, HbA1c, Chol, TG, HDL, LDL, BMI | Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, **VLDL**, BMI |
| Missing | HbA1c 9.3% (portable assay), BMI 1.9%, lipids/renal ≤0.9% | none (FBS absent as a column) |
| Source | Provided by the paper's authors as the "dataset of diabetes research" (raw CSV + Supplementary dictionary). The public repository behind it could not be re-verified with a stable DOI/license at packaging time; we therefore redistribute **only the derived 11-column table** in this repo and ask users to obtain the raw cohort from the original source. | Mendeley Data `wj9rwkp9c2` v1 — <https://data.mendeley.com/datasets/wj9rwkp9c2/1>, **CC BY 4.0** (license verified on the dataset page, 2026-09). |

Notes:

- The paper's *publicly available* dataset (cited in its Data-availability
  section) is IPDD = our **Dataset2**. The paper admits its public copy lacks
  **FBS**, which they obtained privately from the dataset's custodian — so exact
  numeric reproduction of the paper's tables on the public file is impossible by
  construction.
- **Dataset1** (Chinese, n=5922) is our *generality check* for the methodology —
  the paper does not publish results on it.
- Column mapping of the raw Chinese file (README sheet of the author's
  `Dataset1_Paper11_Clear.xlsx`): `Urea←BUN`, `Cr←SCRE`, `FBS←FPG`,
  `Chol←CHOL`, `CLASS←PGclass3` (0=NGR→N, 1→P, 2=DM→Y, per ADA glucose criteria).
  Gender coding in the data is 1=Male / 2=Female (its shipped dictionary says
  0=F/1=M; the mapping used here follows the observed values and the paper's
  convention). This assumption is stated explicitly in `pipeline/outputs/dataset1_audit.json`
  context and in the report.
- The 100%-empty feature column of each dataset (FBS for dataset2, VLDL for
  dataset1) is dropped at load time, leaving 11 usable features each — the two
  feature sets are *not identical* across datasets.

## License / redistribution

- Dataset2 (IPDD, CC BY 4.0): the derived `pipeline/data/dataset2_standard.csv`
  keeps attribution to the original Mendeley record. If you reuse it, cite the
  original dataset and link the CC BY license.
- Dataset1: derived table included for reproducibility; raw cohort is **not**
  bundled here. Obtain it from the original custodians before redistribution.
