# Step 4 — Preprocessing, Outlier Detection & the MICE Imputation Experiment

**Project:** Predicting Israeli High School Bagrut Success Using Socioeconomic Data
**Authors:** Yousef Shehade & Shada Esawi

> Self-contained milestone folder. Step 4 fulfils the Presentation 3+4 data-
> cleaning / statistics slides: the lecturer's **MICE imputation experiment**,
> **outlier detection** (Isolation Forest + LOF), and baseline answers to the two
> **core exploratory questions**. Output: a modeling-ready cache for Step 5.

---

## 1. Directory structure

```
step_4_preprocessing_outliers_imputation_experiment/
├── README.md
├── config.yaml                       # paths, seed, task parameters
├── code/
│   ├── io_load.py                    # load Step-3 table + derived cols (combined grade, logs)
│   ├── imputation_experiment.py      # Task A — MICE vs median
│   ├── outliers.py                   # Task B — Isolation Forest + LOF
│   ├── exploratory.py                # Task C — Q1 resilience, Q2 overachievers
│   └── run_step4.py                  # orchestrator + console summary
├── data/
│   └── cleaned_modeling_ready.csv    # 3,697 × 27 (cluster-present, outlier-flagged)
└── graphs/
    ├── imputation_success_comparison.png
    ├── outlier_detection_mapping.png
    ├── subject_resilience_gap.png
    └── low_ses_overachievers_profile.png
```

Run: `python code/run_step4.py` (paths anchored to the step folder; CWD-independent).

---

## 2. Task A — MICE imputation experiment

Per the guideline *"if no missing values, artificially remove 5–10 % and compare
imputation success"*. We masked **8 %** of the fully-populated CBS `index_value`
(296 of 3,697 cells, `seed=42`), then reconstructed it two ways and scored the
masked cells against ground truth.

| Method | RMSE | MAE | R² |
|---|--:|--:|--:|
| **MICE** (`IterativeImputer`, BayesianRidge) | **0.209** | **0.143** | **0.949** |
| Median baseline | 0.928 | 0.782 | −0.003 |

**MICE cuts RMSE by 77.4 %.** It works because `IterativeImputer` regresses the
masked feature on the others (`index_value` correlates 0.97 with `cluster`),
whereas the median ignores all structure and collapses every gap onto one value.
`graphs/imputation_success_comparison.png` shows it: the MICE density overlays the
original almost perfectly while the median imputation injects a spike at the
median (left), and on the masked cells MICE hugs the identity line while the
median predictions form a flat band (right).

> Methodological note: this experiment is deliberately run on a **complete
> feature**, never on the targets. The grade targets are left missing where the
> source suppressed them (small-cohort privacy) — see Steps 1 & 3.

---

## 3. Task B — Outlier detection (Isolation Forest vs LOF)

Both detectors ran on the same 7-feature standardised space
(`combined_avg_grade`, Math/English 5-unit participation, `cluster`,
`index_value`, `log_population`, `log_total_takers`) over the **3,367**
complete-case school-years, `contamination = 0.05`.

| Detector | Flagged | |
|---|--:|---|
| Isolation Forest (global, tree-isolation) | 169 | |
| Local Outlier Factor (local density, k=20) | 169 | |
| **Consensus (both)** | **35** | Jaccard = 0.12 |
| Flagged by either | 303 | iso-only 134, lof-only 134 |

The **low overlap (Jaccard 0.12)** is the interesting result: the two methods
encode different notions of "anomalous". Isolation Forest catches globally
extreme profiles; LOF catches schools that are odd *relative to their local
neighbourhood*. We therefore keep a conservative **consensus flag**
(`outlier_consensus`, 35 rows) rather than dropping everything either model
dislikes. `graphs/outlier_detection_mapping.png` maps them in SES-vs-grade space.

**Outliers are flagged, not deleted** — `cleaned_modeling_ready.csv` keeps every
row plus the flags, so Step 5 can choose whether to exclude them.

---

## 4. Task C — baseline answers to the two exploratory questions

### Q1 — Which subject is most resilient to socioeconomic disparity?
Gap between **cluster 2** and **cluster 9** (smaller gap ⇒ more resilient):

| Subject | Grade c2→c9 | Grade gap | Cohen's d | 5-unit part. gap |
|---|---|--:|--:|--:|
| **Math** | 78.6 → 84.7 | **+6.18 pts** (+7.9 %) | **0.91** | **+0.115** |
| English | 79.9 → 86.3 | +6.45 pts (+8.1 %) | 1.16 | +0.367 |

**Answer: Mathematics is the more resilient subject.** Its grade gap is smaller
(6.18 vs 6.45 pts, and a smaller effect size d = 0.91 vs 1.16), and its
advanced-track participation is far less socioeconomically stratified
(+0.115 vs +0.367 — English advanced selection is ~3× more SES-sensitive).
Intuition: Math 5-unit is gated by ability/pipeline fairly uniformly, whereas
English 5-unit uptake scales steeply with the wealth/exposure of the locality.
(Caveat: cluster 9 has only 41 school-years, so treat the elite tail as
indicative.) Plot: `graphs/subject_resilience_gap.png`.

### Q2 — Are there low-SES overachievers, and how do they select subjects?
Schools were aggregated to **school level (mean across years)** so "consistently"
is captured. Benchmark = **median combined grade of elite (cluster 8–10) schools
= 84.90**.

- **87 of 460** low-SES (cluster 1–4) schools (**18.9 %**) match or exceed the
  elite median grade — a substantial resilient group, not a handful.
- Their **advanced-track selection is dramatically higher** than ordinary low-SES
  schools:

  | 5-unit participation | Overachievers | Normal low-SES | p-value |
  |---|--:|--:|--:|
  | Math | 0.132 | 0.049 | 1.4 × 10⁻⁵ |
  | English | 0.468 | 0.181 | 1.4 × 10⁻¹¹ |

  i.e. they push **~2.7× more students into advanced Math** and **~2.6× more into
  advanced English** — they channel pupils into the hard tracks across the board.
- The top names are **selective / religious / science-oriented** schools
  (e.g. *תיכון ליד האוניברסיטה*, *מעלות בית יעקב*, *עתיד למדעים לוד*), suggesting
  selection and programme focus, not locality wealth, drive their results.

Plot: `graphs/low_ses_overachievers_profile.png`.

---

## 5. Output — `data/cleaned_modeling_ready.csv` (3,697 × 27)

The Step-3 school-year table (cluster-present rows only) **plus**: derived
features (`combined_avg_grade`, `total_takers`, `log_population`,
`log_total_takers`) and outlier metadata (`iso_outlier`, `lof_outlier`,
`iso_score`, `lof_score`, `outlier_consensus`, `outlier_any`). Targets keep their
NaNs (never imputed).

---

## 6. Mandated-method coverage (lecturer's list)

| Method | Used in | 
|---|---|
| **MICE** (IterativeImputer) | Task A imputation experiment |
| **Isolation Forest** | Task B outlier detection |
| **Local Outlier Factor (LOF)** | Task B outlier detection |

(SGD / Boruta / SHAP are reserved for the Step 5 modeling phase.)

---

## 7. Step 4 verification checklist

- [x] MICE vs median experiment on a complete feature (8 % masked) — MICE −77 % RMSE.
- [x] Imputation density/scatter plot saved.
- [x] Isolation Forest + LOF run, contrasted (169 / 169, consensus 35, Jaccard 0.12).
- [x] Outlier mapping plot saved; outliers flagged, not dropped.
- [x] Q1 resilience gap computed + plotted → **Math more resilient**.
- [x] Q2 overachievers identified (87/460, 18.9 %) + profiled + plotted.
- [x] `cleaned_modeling_ready.csv` (3,697 × 27) written; targets not imputed.

**Status: Step 4 complete ✔ — awaiting signal to begin Step 5.**
