# Predicting Israeli High School Bagrut Success Using Socioeconomic Data

**Data Science Lab — Final Project**
**Authors:** Yousef Shehade & Shada Esawi

> This README documents **Step 1 — Ingestion & Text Standardisation** only.
> Later steps (merging, imputation, outlier handling, modelling, explainability)
> will be appended as they are built.

---

## 1. What Step 1 accomplishes

Step 1 turns two messy raw files into clean, well-typed, **text-standardised**
intermediate tables and produces four diagnostic plots that we inspect *before*
merging. Concretely, it:

1. **Ingests** both datasets resiliently (BOM handling, Excel header offset,
   placeholder coercion).
2. **Standardises Hebrew text** keys so the datasets can later be joined on
   locality name (the numeric `semel` codes are **not** compatible — see below).
3. **Caches** the cleaned tables to `outputs/data/`.
4. **Visualises** four data characteristics to `outputs/graphs/`.
5. **Prints a verification summary** of shapes, missingness and join coverage.

Nothing is merged or modelled yet — that is Step 2 onward.

---

## 2. Directory structure

Each phase of the project lives in its own self-contained `step_*` folder. The
raw `datasets/` are shared at the project root.

```
BagrutProject/
├── datasets/                            # SHARED raw inputs (read-only)
│   ├── israel_bagrut_averages.csv
│   └── downloadFile.xlsx
└── step_1_ingestion_standardization/    # ← this self-contained step
    ├── README.md                        # this file (Step 1 focus)
    ├── config.yaml                      # config scoped to this step (raw data via ../datasets/)
    ├── code/                            # CODE SECTION
    │   ├── io_load.py                   # resilient ingestion (BOM, Excel header=10, '..' -> NaN)
    │   ├── clean_text.py                # Hebrew normalisation + caches cleaned CSVs
    │   ├── visualize.py                 # the four Step-1 plots
    │   └── run_step1.py                 # orchestrator: load -> clean -> visualise -> report
    ├── data/                            # INTERIM DATA SECTION (cleaned CSVs)
    │   ├── bagrut_clean.csv
    │   └── ses_clean.csv
    └── graphs/                          # GRAPHS SECTION (PNG plots)
        ├── raw_grade_distribution.png
        ├── studyunits_breakdown.png
        ├── ses_cluster_frequency.png
        └── target_missingness.png
```

Path resolution is anchored to this step folder (each module computes it from its
own location), so the scripts run correctly from any working directory. The
shared raw datasets are referenced as `../datasets/...` in `config.yaml`.

---

## 3. Raw file inputs

> **External data sources** (the raw files are git-ignored, not redistributed in
> this repository — download them to `datasets/` to reproduce):
> - **Dataset 1 — Bagrut Grades 2013–2016** (released under the Israeli Freedom of
>   Information Law), hosted on Kaggle:
>   <https://www.kaggle.com/datasets/emachlev/bagrut-israel/data>
> - **Dataset 2 — CBS Socioeconomic Index**: official publication of the Israel
>   Central Bureau of Statistics (CBS / הלשכה המרכזית לסטטיסטיקה),
>   <https://www.cbs.gov.il/>.

### Dataset 1 — `datasets/israel_bagrut_averages.csv` (Bagrut grades 2013–2016)
- **69,638 rows × 8 columns**, encoding **`utf-8-sig`** (UTF-8 **with BOM**).
- Grain: one row = *school × subject × study-units × year*.
- Columns: `grade` (target), `takers`, `studyunits` (2/3/4/5; **5 = advanced track**),
  `year`, `subject`, `city`, `school`, `semel` (**school** code).
- All Hebrew text values are **padded with trailing spaces** (100 % of `city`).
- `grade` is **21.4 % missing** — by design (see §6).

### Dataset 2 — `datasets/downloadFile.xlsx` (CBS socioeconomic index)
- Sheet **`גיליון1`**, **1,208 locality rows** after cleaning.
- Columns (renamed to canonical English in code):
  `locality_code` (`סמל היישוב` — **locality** code), `locality_name` (`שם יישוב`),
  `locality_form` (`צורת יישוב`), `population` (`אוכלוסייה`),
  `index_value` (`ערך מדד`), `cluster` (`אשכול (מ-1 עד 10)`, 1–10).

> **Key incompatibility:** the `semel` in Dataset 1 is a *school* code while
> `סמל היישוב` in Dataset 2 is a *locality* code — they share **0** values. The
> datasets must therefore be joined on **locality name**, which is why Step 1
> invests heavily in Hebrew text standardisation.

---

## 4. CBS file row-parsing offsets

The workbook has leading blank/metadata rows:

| Physical row index | Content                                   |
|--------------------|-------------------------------------------|
| 0 – 9              | blank / report metadata (ignored)         |
| **10**             | **real column headers** (`header_row: 10`)|
| 11                 | blank spacer row (dropped)                |
| **12 …**           | locality data records                     |

Ingestion reads with `header=10`, then `dropna(how="all")` removes the blank
spacer and any trailing empties. The unranked-locality placeholder token
**`..`** (25 localities) is coerced to `NaN`, and `population` / `index_value` /
`cluster` are cast to real numbers.

---

## 5. Hebrew text-normalisation rules (`clean_text.py`)

Applied to the locality keys (`city` → `city_norm`, `locality_name` →
`locality_norm`); the original columns are preserved.

1. **Strip parenthetical qualifiers** — `(...)` groups and stray bracket chars
   (e.g. `(יישוב)`, `(קבוצה)`, `(איחוד)`, `(מוסד)`).
2. **Remove quote marks** — gershayim/geresh and ASCII quotes (`"`, `'`, `״`, `׳`).
3. **Collapse whitespace** — trim ends + squeeze internal runs to a single space
   (removes the trailing-space padding).
4. **Yod-doubling prefix fix** — `קרית` → `קריית` (word-initial only).
5. **Documented spelling map** — exact whole-string variants, e.g.
   `הרצליה → הרצלייה`, `נהריה → נהרייה`, `דבוריה → דבורייה`, `יהוד → יהוד מונוסון`,
   `אפרתה → אפרת`.

These rules raise the exact normalised-key overlap from **87.0 %** (raw) to
**92.8 %** of Bagrut records. The remaining ~7 % (merged municipalities,
additional spelling variants, and genuinely non-municipal *regional/boarding*
schools) are resolved in **Step 2** via fuzzy matching (`rapidfuzz`) + a small
manual crosswalk. They are intentionally **not** force-matched here.

---

## 6. Note on the 21 % missing target (`grade`)

The missingness is **not random**: where `grade` is missing the median cohort is
**7** test-takers; where present it is **28** (the `takers` floor is 5). The
source **suppresses averages for small cohorts** for privacy (MAR/censoring).
Plot 4 validates this visually. **Implication for later steps:** we will *not*
impute the target; instead we build the school-level target from observed grades
weighted by `takers`, and run the mandated "remove 5–10 % then impute & compare"
experiment on a *complete feature* (e.g. `index_value`/`population`) rather than
on the censored label.

---

## 7. How to run

```bash
# From inside this step folder (uses the Anaconda Python 3.11 env):
cd step_1_ingestion_standardization
python code/run_step1.py
```

The scripts are path-independent, so this also works from the project root:
`python step_1_ingestion_standardization/code/run_step1.py`.

Individual modules are also runnable for debugging:
`python code/io_load.py`, `python code/clean_text.py`, `python code/visualize.py`.

**Dependencies:** `pandas`, `numpy`, `openpyxl`, `pyyaml`, `matplotlib`,
`seaborn` (all present in the Anaconda env). `rapidfuzz`, `shap`, `boruta` are
deferred to later steps.

---

## 8. The four Step-1 plots (`outputs/graphs/`)

| File | What it shows |
|------|----------------|
| `raw_grade_distribution.png` | Histogram + KDE of observed `grade` — confirms a near-normal shape (mean 79.1, median 79.8). |
| `studyunits_breakdown.png`   | Record counts per study-unit level — 5-unit dominates (32,292). |
| `ses_cluster_frequency.png`  | Localities per socioeconomic cluster — clear skew toward clusters 6–8. |
| `target_missingness.png`     | `takers` for missing- vs present-grade cells (log scale) — validates small-cohort suppression. |

---

## 9. Step 1 verification checklist

- [x] Bagrut CSV loads with `utf-8-sig`; first header is `grade` (no BOM artefact).
- [x] Bagrut shape = **(69,638, 8)** raw → **(69,638, 9)** with `city_norm`.
- [x] CBS workbook parsed with `header=10`; shape = **(1,208, 7)** with `locality_norm`.
- [x] `..` placeholders coerced to `NaN` (**25** unranked localities); numeric cols typed.
- [x] Trailing-space padding removed: **69,638 → 0** padded city values.
- [x] Hebrew normalisation applied; normalised-key overlap = **92.8 %** of records.
- [x] Cleaned tables cached to `outputs/data/` (`bagrut_clean.csv`, `ses_clean.csv`).
- [x] **4/4** plots written to `outputs/graphs/`.
- [x] `run_step1.py` prints the verification summary and exits cleanly.

**Status: Step 1 complete ✔ — awaiting signal to begin Step 2 (Merging).**
