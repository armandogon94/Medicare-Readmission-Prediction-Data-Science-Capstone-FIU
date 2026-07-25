# Data Access

This project is built on **MIMIC-IV v3.1**, a de-identified critical-care database
released by the MIT Laboratory for Computational Physiology via PhysioNet.

**No MIMIC-IV data — raw or derived — is distributed in this repository.**
MIMIC-IV is governed by the PhysioNet Credentialed Health Data Use Agreement, which
restricts access to individually credentialed researchers and prohibits
redistribution, including of derived tables. Every table this project consumes must
be built from your own credentialed copy.

## Getting access

1. Create a PhysioNet account — <https://physionet.org/register/>
2. Complete the **CITI "Data or Specimens Only Research"** training course and
   upload your completion report to your PhysioNet profile.
3. Sign the credentialed-access DUA for **MIMIC-IV v3.1** —
   <https://physionet.org/content/mimiciv/3.1/>
4. Approval typically takes a few business days.

## Building the tables

Once approved, download MIMIC-IV v3.1 and place the source tables where the
notebook expects them:

```
Dataset/
└── mimic-parquet/
    ├── training_table_v1.parquet     # 21-feature baseline cohort
    ├── training_table_v10.parquet    # 184-column expansion set
    ├── admissions.parquet            # raw MIMIC-IV admissions
    └── omr.parquet                   # raw MIMIC-IV outpatient measurements
```

Then open `Capstone_Final_Notebook.ipynb` and run it top to bottom:

- **Section 8.2** rebuilds `training_table_v7.parquet` (the 50-feature parsimonious
  set) and `v7_split.npz` from `training_table_v10` joined against the raw
  `admissions` and `omr` tables — target encodings, PCA components, interaction
  terms, log transforms, and the stratified train/test split.
- The cell is idempotent: it regenerates the artifacts if they are absent and loads
  them from disk if they already exist. Expect 30–60 seconds on first run.
- `RANDOM_STATE = 42` and `TEST_SIZE = 0.20` are fixed, so the split is reproducible.

Everything downstream — Optuna tuning, the 4-GBM ensemble, the Nelder-Mead blend,
the SHAP analysis, and every figure in the report — runs from those regenerated
artifacts.

## Cohort definition

The cohort is the Medicare-insured subset of MIMIC-IV v3.1: **244,576 admissions**,
**21.1%** positive rate for unplanned 30-day readmission. Cohort construction is
documented in Section 5 of the notebook.

## Why the data is not vendored

Committing the derived training table would violate the DUA even though MIMIC-IV is
de-identified under HIPAA Safe Harbor — de-identification governs *re-identification
risk*, not *redistribution rights*. Keeping the pipeline reproducible while shipping
none of the data is the correct handling for restricted health datasets, and it is
the standard this repository holds itself to.
