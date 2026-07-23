# Step 5 — Predictive Modeling, Ablation & Explainability

**Project:** Predicting Bagrut Success from Municipal Socioeconomics and School-Level Institutional Resources
**Authors:** Yousef Shihade & Shada Esawi

> The modelling stage. Collinearity is handled by **iterative VIF pruning**,
> features are chosen per target by **Boruta** over a 49-column SES+budget
> candidate space, four model families compete under **GroupKFold(`semel`)**
> cross-validation, and the champion is tuned and explained with **SHAP**. A
> dedicated **ablation study** then isolates how much the institutional data
> contributes over municipal socioeconomics alone — the study's central
> quantitative claim.

---

## 1. Directory structure

```
step_5_predictive_modeling_explainability/
├── README.md
├── config.yaml              # candidate features, VIF threshold, Boruta params, ablation & display labels
├── code/
│   ├── io_load.py            # load Step-4 data; translate Hebrew categoricals to English; build X/y/groups
│   ├── feature_selection.py  # iterative VIF pruning + Boruta selection
│   ├── modeling.py           # 4-model tournament + tuned HGB champion
│   ├── ablation.py           # SES-only vs Boruta-selected full set, same rows
│   ├── explain.py            # SHAP, leaderboard, ablation & VIF plots
│   └── run_step5.py          # orchestrator + comprehensive console report
├── models/                   # 4 tuned HGB models + VIF/Boruta/ablation/leaderboard CSVs
│                             #   (leaderboard_cv = untuned tournament;
│                             #    leaderboard_tuned = headline champion numbers)
└── graphs/                   # VIF pruning, 4 SHAP beeswarms, leaderboard, ablation chart
```

Run: `python code/run_step5.py`.

---

## 2. Collinearity — iterative VIF pruning

With 15 numeric candidates, redundancy can hide anywhere, and checking pairs by
hand does not scale. This step therefore **repeatedly** computes VIF, drops the
single worst offender, and **recomputes** — because dropping one feature can
resolve another's inflation, which a naive one-pass cutoff would miss entirely.

![VIF pruning](graphs/vif_pruning.png)

| Step | Dropped | VIF at drop |
|---|---|--:|
| 1 | `teaching_budget_per_student` | 76.57 |
| 2 | `total_budget_per_student` | 26.28 |
| 3 | `index_value` | 20.59 |

**12 numeric features survive.** The chart shows *why* iteration matters:
`cluster` starts at VIF 20.0 (would look collinear on a naive single pass) but
**survives**, because its inflation was entirely caused by `index_value` — once
that's dropped, cluster's recomputed VIF falls well under the threshold. This
correctly identifies `cluster ↔ index_value` as a **mutual** redundant pair and
keeps the more interpretable of the two, alongside two redundant budget pairs
that no manual inspection had anticipated.

---

## 3. Feature selection — Boruta on the full SES+budget space

Boruta ran **once per target** on the 12 VIF-surviving numeric features + 7
categoricals (locality_form, district, sector, supervision, legal_status,
education_stage, **year**) — 49 encoded candidate columns in total.

### Why `year` is treated as categorical

Feeding `year` (2013–2016) as a plain number would implicitly assume each year
shifts the outcome by a fixed, linear amount. With only four discrete exam
periods there is no basis for that: nothing says the 2013→2014 change should
equal the 2015→2016 change, and imposing linearity discards that flexibility for
nothing in return. `year` is therefore **one-hot encoded into 4 independent
columns** (`year_2013` … `year_2016`) — each period gets its own effect, the
same treatment every other categorical (sector, district, …) receives.

**Result: no individual year-dummy is confirmed by Boruta for any target.**
Tested independently, none clears the relevance bar — evidence that whatever
weak year-related signal exists is better described as a **mild, gradual drift**
across 2013–2016 than as any single anomalous year. Every headline result below
is therefore free of a time-trend assumption.

**A 10-feature core is confirmed for all four targets:** `cluster`,
`log_population`, `nurture_quintile`, `avg_class_size`, `log_school_size`, and
the five per-student budget ratios (`tuition`, `perimeter`, `projects`,
`purchases`, `transport`). On top of that shared core, each target confirms its
own additions:

| Target | Additional features beyond the 10-feature core | Total |
|---|---|--:|
| `math_avg_grade` | **supervision_Haredi** | **11** |
| `math_5unit_participation` | **district_North**, **sector_Jewish** | **12** |
| `english_5unit_participation` | private_hours_per_student, **sector_Jewish**, **supervision_Haredi** | **13** |
| `english_avg_grade` | private_hours_per_student, special_ed_share, **district_North**, **sector_Bedouin** | **14** |

Note that the extras are **genuinely target-specific, not cumulative** — e.g.
`supervision_Haredi` is confirmed for `math_avg_grade` but *not* for
`english_avg_grade`, and `district_North` is confirmed for
`math_5unit_participation` but *not* for `english_5unit_participation`. Both
English targets pick up `private_hours_per_student` (private tutoring hours),
which neither Math target does.

**Five budget ratios are confirmed for every single target**
(`tuition_per_student`, `perimeter_per_student`, `projects_per_student`,
`purchases_per_student`, `transport_per_student`) — a much richer, more stable
selection story than municipal features alone can support. Boruta also
confirms specific **sector/supervision/district** dummies per target — school
structural identity carries real, independent signal beyond municipal cluster.

Full per-target detail: `models/boruta_report.csv`.

---

## 4. Modeling tournament & tuned champion

GroupKFold(`semel`) throughout, a 4-model tournament, then HGB tuned via
RandomizedSearchCV — run on each target's Boruta-selected features.

![leaderboard](graphs/models_performance.png)

### Final leaderboard — tuned HistGradientBoosting

| Target | R² | RMSE | MAE |
|---|--:|--:|--:|
| **english_5unit_participation** | **0.545** | 0.178 | 0.129 |
| `math_avg_grade` | 0.431 | 5.273 | 4.059 |
| `english_avg_grade` | 0.428 | 4.584 | 3.502 |
| `math_5unit_participation` | 0.421 | 0.079 | 0.056 |

These headline numbers are saved to **`models/leaderboard_tuned.csv`** (and
inside each model's `.joblib` as `cv_metrics`). Note the distinction:
`models/leaderboard_cv.csv` holds the **untuned 4-model tournament**, which is
what the chart above plots; `leaderboard_tuned.csv` holds the **tuned champion**
quoted here and in the root README.

All four untuned models score **positive R² across the board** — even
RandomForest, the most overfit-prone of them on a narrow feature space, holds up
here. The breadth of the feature set gives every model family real signal to
work with.

---

## 5. 🎯 Ablation study — does the budget dataset add information beyond SES?

This is the study's central experiment. For every target we tune
HistGradientBoosting **twice on IDENTICAL rows**: once on municipal features
only (`cluster`, `log_population`, `locality_form`, and `year` — one-hot, see
§3 — the **"SES only"** arm) and once on whatever Boruta
selected from the full SES+budget space (the **"SES + Budget"** arm). Same
rows, same GroupKFold folds, same tuning protocol — so the R² delta is
attributable **only** to the extra information.

![ablation](graphs/ablation_before_after.png)

| Target | SES only | **SES + Budget** | **ΔR²** |
|---|--:|--:|--:|
| `math_avg_grade` | 0.138 | **0.458** | **+0.320** |
| `english_avg_grade` | 0.199 | **0.455** | **+0.256** |
| `math_5unit_participation` | 0.058 | **0.439** | **+0.381** |
| `english_5unit_participation` | 0.229 | **0.549** | **+0.321** |

> **Why these R² values differ slightly from §4's leaderboard** (e.g.
> `math_avg_grade`: 0.458 here vs 0.431 there). The ablation deliberately
> restricts both arms to the **row intersection where *both* feature sets are
> complete** (2,929–3,187 rows depending on target, vs the full sample §4 uses),
> because a fair before/after comparison requires identical rows. It also
> re-expands any Boruta-selected categorical back to its **full** dummy set, so
> the "after" arm carries a few more encoded columns than §4's model. The §4
> leaderboard is therefore the number to quote for **model performance**; the
> ΔR² here is the number to quote for **how much the budget data adds**.

**Mean ΔR² across the four targets: +0.320.** Every target's explanatory power
**more than doubled** (and `math_5unit_participation` grew nearly **8×**). The
gain comes not from the budget ratios alone but from the school-level
**sector, supervision, and district** attributes alongside them, all of which
Boruta confirms carry real, independent predictive signal.

---

## 6. SHAP explainability

![SHAP example](graphs/shap_beeswarm_math_5unit_participation.png)

For `math_5unit_participation` — the target municipal SES explains least
— the top SHAP features are `nurture_quintile`, `log_school_size`, and
`transport_per_student`, with `district_North` and `avg_class_size` also
ranking above `cluster`. **Institutional/school-level attributes outrank
municipal wealth** for explaining who enters advanced Math — direct, visual
confirmation of the ablation result.

---

## 7. Headline answer to the research question

**Municipal socioeconomic status alone is a weak-to-moderate predictor**
(R² 0.06–0.23). **Adding school-level institutional resources — budget, class
size, sector, supervision, district — roughly triples explanatory power**
(R² 0.42–0.55). The variance municipal SES cannot explain is not noise: a
large share of it is **institutional and structural school identity**, which
this pipeline captures from the start rather than as an afterthought.

---

## 8. Step 5 verification checklist

- [x] Iterative VIF pruning run on the full 15-candidate numeric set; 3 dropped
      (2 new budget redundancies + the known cluster/index_value pair), with the
      mutual-pair logic demonstrated visually.
- [x] `year` treated as 4 one-hot categories, not an assumed linear trend;
      Boruta confirms none individually — headline results unchanged.
- [x] Boruta run per target on the full 49-column SES+budget space; ≥11 features
      confirmed per target.
- [x] 4-model tournament + tuned HGB champion; GroupKFold(semel) throughout.
- [x] Ablation study: SES-only vs Boruta-selected full set, identical rows,
      identical protocol; mean ΔR² = +0.320.
- [x] SHAP beeswarms for all 4 targets; Hebrew categorical labels translated to
      English for readability (matplotlib RTL rendering issue caught and fixed).
- [x] 4 tuned models + VIF/Boruta/ablation/leaderboard CSVs saved; tuned champion
      metrics persisted to `leaderboard_tuned.csv` so every headline number in the
      READMEs is auditable from a plain CSV.

**Status: Step 5 complete ✔**
