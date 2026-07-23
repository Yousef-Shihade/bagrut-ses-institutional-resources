# Step 3 — Feature Engineering & Target Setup

**Project:** Predicting Bagrut Success from Municipal Socioeconomics and School-Level Institutional Resources
**Authors:** Yousef Shihade & Shada Esawi

> This step re-grains the merged record-level table to **one row per school per
> year**, derives the **4 modelling targets**, and builds the predictor space:
> the municipal CBS features plus **8 per-student budget ratios** and **5
> school-level categoricals** engineered from the Ministry data — **23 candidate
> features** in total.

---

## 1. Directory structure

```
step_3_feature_engineering_target_setup/
├── README.md
├── config.yaml                # grain, subject defs, feature/ratio definitions
├── code/
│   ├── io_load.py
│   ├── feature_engineering.py # subject aggregation + budget ratio engineering
│   ├── visualize.py           # 4 plots
│   └── run_step3.py           # orchestrator + verification summary
├── data/
│   └── school_level_features_targets.csv   # 3,731 × 45
└── graphs/
```

Run: `python code/run_step3.py`.

---

## 2. Design decisions

| Decision | Rationale |
|---|---|
| **Targets: Math + English only** | Deliberate scope discipline — these two subjects are taken near-universally, so their grades and 5-unit rates are comparable across schools. Niche electives would introduce heavy selection bias (see the Step 2 discussion). |
| **Grain: `semel × year`** | The research question is about *schools*, not individual exam sittings. One row per school-year keeps every year's observation while matching the unit of analysis. |
| **Grade targets use units 3/4/5; participation uses 5-unit** | Grades average over all academic-track sittings; participation isolates the advanced track specifically. |
| **`log_population` computed here, not in preprocessing** | A deterministic transform of a raw feature is feature engineering, not data cleaning — keeping it here means the preprocessing step handles only imputation and outliers. |
| **Targets are never imputed** | Missing grades are privacy suppression of small cohorts (Step 1), so imputing them would fabricate the outcome. Aggregation runs over observed cells only. |

---

## 3. The 4 targets

| Target | Non-null | Mean |
|---|--:|--:|
| `math_avg_grade` | 3,292 (88.2%) | 78.68 |
| `english_avg_grade` | 3,280 (87.9%) | 80.81 |
| `math_5unit_participation` | 3,668 (98.3%) | 0.087 |
| `english_5unit_participation` | 3,688 (98.8%) | 0.325 |

**3,731 school-years across 1,022 schools.** The two participation targets are
near-complete (98%+) because they are computable whenever a school had any
academic-track takers; the grade targets carry the ~12% suppression documented
in Step 1.

---

## 4. The candidate feature space (23 features)

![feature inventory](graphs/feature_inventory.png)

| Group | Features | Count |
|---|---|--:|
| CBS municipal | `cluster`, `index_value`†, `population`→`log_population`, `locality_form` | 4 |
| Budget categorical | `district`, `sector`, `supervision`, `legal_status`, `education_stage` | 5 |
| Budget direct numeric | `nurture_quintile`, `avg_class_size` | 2 |
| **Budget engineered ratios** | `total_budget_per_student`, `teaching_budget_per_student`, `tuition_per_student`, `perimeter_per_student`, `projects_per_student`, `purchases_per_student`, `transport_per_student`, `private_hours_per_student` | 8 |
| Budget derived | `special_ed_share`, `log_school_size` | 2 |
| Temporal | `year` | 1 |
| **Total** | | **23** |

† `index_value` is carried through so Step 5's VIF analysis can quantify its
redundancy with `cluster` (r ≈ 0.97), after which it is dropped.

All 8 ratios use the same guard pattern — `numerator/denominator` with
`students_regular ≤ 0` and `±inf` forced to `NaN`, never silently propagated.
Coverage: **98.4%** across every ratio (3,672/3,731 school-years).

---

## 5. ⚠️ Data-quality note — `transport_per_student` can be negative

`תקציב הסעות` (transport budget) is **negative for ~86% of institutions** in the
raw Ministry file (min −₪33,722; median per-student **−₪7.15**). This is not a
bug: it is almost certainly a **net budget adjustment/correction** figure rather
than gross transport spending. We kept the column **as-is** — the sign may still
carry real signal — but flag it here so it is never mis-read as a plain cost.

---

## 6. Budget ratios are independent of municipal cluster (verified)

![correlation](graphs/budget_ratio_correlation.png)

Every engineered budget ratio correlates **near-zero with `cluster`** (all
|r| ≤ 0.09) — confirming these carry genuinely new, independent information
rather than acting as a disguised proxy for municipal wealth. This is the
central justification for adding the Ministry dataset at all. (Contrast with
`index_value`, r ≈ 0.97 with cluster — redundant, and dropped in Step 5.)

Note: this is about the **numeric ratios**. Step 2 already showed the
**categorical** `sector` *is* associated with cluster (Arab/Bedouin/Druze schools
concentrate in low clusters) — so categoricals and ratios behave differently and
both deserve attention in Step 5's collinearity analysis.

---

## 7. Plots (`graphs/`)

| File | Shows |
|---|---|
| `target_distributions.png` | histograms of all 4 targets |
| `cluster_vs_targets.png` | Math/English avg grade by cluster (SES gradient) |
| `feature_inventory.png` | candidate feature count by source (23 total) |
| `budget_ratio_correlation.png` | budget ratios × cluster correlation heatmap |

---

## 8. Step 3 verification checklist

- [x] Filtered to Math + English only (24.4% of subject-cells); units 3/4/5 for
      grade, 5-unit for participation.
- [x] Re-grained to `semel × year`: 3,731 rows, 1,022 schools.
- [x] Targets computed from observed cells only — never imputed.
- [x] 8 budget ratios engineered with robust div-by-zero/inf guards; 98.4% coverage.
- [x] 5 school-level categoricals + `nurture_quintile` + `avg_class_size` carried forward.
- [x] `transport_per_student` sign anomaly investigated and documented, not hidden.
- [x] Budget ratios confirmed near-orthogonal to cluster (|r| ≤ 0.09) — genuine
      new information, verified across the full ratio set.
- [x] 4/4 plots saved.
- [x] `school_level_features_targets.csv` (3,731 × 45) written.

**Status: Step 3 complete ✔**
