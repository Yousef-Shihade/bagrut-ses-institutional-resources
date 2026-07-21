# 📦 ARCHIVE — v1: Two-Dataset Pipeline (superseded)

**Project:** Predicting Israeli High School Bagrut Success
**Authors:** Yousef Shihade & Shada Essawi
**Status:** ⚠️ **Superseded** by the v2 three-dataset pipeline at the repository root.
**Archived:** 2026-06-21

---

## Why this is archived

This was the **original pipeline**, built around a two-dataset research question:

> *"Can a municipality's socioeconomic status **alone** predict a school's Bagrut
> success?"*

Datasets 1 & 2 (Bagrut grades + CBS socioeconomic index) drove Steps 1–5, and the
Ministry-of-Education **budget dataset was added only at the end, as a bolt-on
Step 6 "extension."**

### Lecturer feedback that triggered the rebuild

1. **The budget dataset should be integrated from the start**, not appended as an
   extra step — it belongs inside the core research question.
2. **Too few features.** Restricting Steps 1–5 to municipal socioeconomic
   variables left only ~4 candidate predictors (`cluster`, `log_population`,
   `year`, `locality_form`).
3. **The MICE experiment needed repeated trials** — a single masked run isn't
   enough evidence; multiple iterations must show a stable, reproducible result.

v2 therefore merges **all three datasets in Steps 1–2**, engineers a far richer
feature space in Step 3, and runs a multi-iteration MICE robustness experiment in
Step 4. Step 6 no longer exists as a separate stage — it is absorbed into 1–5.

---

## What v1 contained

| Stage | Scope |
|---|---|
| `step_1_ingestion_standardization/` | Hebrew text/encoding standardisation (Bagrut + CBS) |
| `step_2_data_merging_integration/` | 4-stage fuzzy locality-name merge (Bagrut ↔ CBS) |
| `step_3_feature_engineering_target_setup/` | Re-grain to school × year; 4 targets |
| `step_4_preprocessing_outliers_imputation_experiment/` | MICE (single run), Isolation Forest + LOF, 2 exploratory questions |
| `step_5_predictive_modeling_explainability/` | VIF, Boruta, model tournament, tuned HGB, SHAP |
| `step_6_budget_integration_performance_boost/` | **Bolt-on** budget integration (the part now folded into v2 Steps 1–5) |

---

## Headline v1 results (preserved for the "before vs after" narrative)

These remain valid as the **v1 baseline** and are useful evidence of improvement:

- **Merge:** 99.44 % of 69,638 Bagrut records matched to a CBS locality.
- **MICE (single run):** R² = 0.949, RMSE 0.209 vs median baseline R² ≈ −0.003.
- **Outliers:** Isolation Forest 169 · LOF 169 · consensus 35 (Jaccard 0.12).
- **Low-SES overachievers:** 87 / 460 = 18.9 % (p ≤ 1e-5).
- **Step-5 baseline (SES only), tuned HGB CV R²:**
  `english_5unit 0.192` · `english_avg 0.150` · `math_avg 0.094` · `math_5unit 0.056`
- **Step-6 (SES + dual budget), tuned HGB CV R²:**
  `english_5unit 0.391` · `english_avg 0.220` · `math_avg 0.208` · `math_5unit 0.211`
- **Boruta:** confirmed `cluster` for the English targets, rejected it for Math —
  independent corroboration that Math is more socioeconomically resilient.

---

## ⚠️ Note on running this code

The modules anchor paths relative to their own step folder (`../datasets/...`,
`../step_4_.../data/...`). After the move into `archive/v1_two_dataset_pipeline/`,
those relative paths **no longer resolve**. This code is retained as a
**reference and audit trail**, not as a runnable pipeline. Use the v2 pipeline at
the repository root instead.
