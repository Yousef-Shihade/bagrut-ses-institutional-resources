"""
run_step3.py — Step 3 orchestrator (v2: feature engineering & target setup).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Flow:
    load Step-2 merged table (118 subjects, all 3 sources)
    -> filter to Math+English, re-grain to school x year
    -> engineer 4 targets + 8 budget ratios + special_ed_share + log_school_size
       + log_population
    -> 4 diagnostic plots + verification summary

Usage:
    python code/run_step3.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_engineering as fe  # noqa: E402
import visualize as viz  # noqa: E402
from io_load import load_config, load_merged, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 78


def main() -> None:
    cfg = load_config()
    graphs = resolve(cfg["paths"]["out_graphs"])

    print(_hr())
    print("STEP 3 — FEATURE ENGINEERING & TARGET SETUP (v2)")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    df = load_merged(cfg)
    print(f"\n[LOAD] Step-2 merged table: {df.shape}  ({df['subject'].nunique()} subjects)")

    core = cfg["subjects"]["core"]
    n_math = (df["subject"] == core["math"]).sum()
    n_eng = (df["subject"] == core["english"]).sum()
    print(f"\n[FILTER] Math ({core['math']}): {n_math:,} rows | "
          f"English ({core['english']}): {n_eng:,} rows | "
          f"= {100*(n_math+n_eng)/len(df):.1f}% of all subject-cells")

    school_level = fe.build_school_level(df, cfg)
    print(f"\n[AGGREGATE] re-grained to semel x year: {school_level.shape[0]:,} rows")
    print(f"    distinct schools: {school_level['semel'].nunique():,}")
    print(f"    years: {sorted(school_level['year'].dropna().unique())}")

    # ---------------- targets ------------------------------------------------ #
    print("\n[TARGETS] (takers-weighted grade / 5-unit participation)")
    for t in ["math_avg_grade", "english_avg_grade",
             "math_5unit_participation", "english_5unit_participation"]:
        s = school_level[t]
        print(f"    {t:30s} non-null {s.notna().sum():5,d} ({s.notna().mean()*100:5.1f}%)  "
              f"mean {s.mean():.3f}")

    # ---------------- feature inventory --------------------------------------- #
    print("\n[FEATURES] candidate predictors carried forward")
    cbs_numeric = [c for c in cfg["cbs_features"] if c != "ses_locality_name"]
    print(f"    CBS municipal ({len(cbs_numeric)}): {cbs_numeric} + log_population")
    print(f"    Budget categorical ({len(cfg['budget_categorical'])}): {cfg['budget_categorical']}")
    ratio_names = list(cfg["budget_ratios"].keys())
    print(f"    Budget direct numeric ({len(cfg['budget_direct_numeric'])}): {cfg['budget_direct_numeric']}")
    print(f"    Budget engineered ratios ({len(ratio_names)}): {ratio_names}")
    print(f"    Budget derived ({2}): ['special_ed_share', 'log_school_size']")
    print(f"    Temporal (1): ['year']")
    total_conceptual = (len(cbs_numeric) + 1 + len(cfg["budget_categorical"])
                        + len(cfg["budget_direct_numeric"]) + len(ratio_names) + 2 + 1)
    print(f"    TOTAL CONCEPTUAL FEATURES: {total_conceptual}  "
          f"(v1 baseline was 4 — Boruta in Step 5 will select among these)")

    # ---------------- ratio engineering coverage ------------------------------ #
    print("\n[BUDGET RATIOS] coverage on the school-year table")
    for name in ratio_names:
        if name in school_level.columns:
            s = school_level[name]
            print(f"    {name:30s} {s.notna().sum():5,d}/{len(s):5,d} "
                  f"({s.notna().mean()*100:5.1f}%)  median={s.median():,.1f}")

    # ---------------- save ------------------------------------------------------ #
    out_path = resolve(cfg["paths"]["school_level_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    school_level.to_csv(out_path, index=False, encoding=cfg["io"]["encoding"])
    print(f"\n[SAVE] {out_path.name}  ({school_level.shape[0]:,} x {school_level.shape[1]})  "
          f"({out_path.stat().st_size/1024/1024:.2f} MB)")

    # ---------------- plots -------------------------------------------------- #
    print("\n[PLOTS]")
    paths = [
        viz.plot_target_distributions(school_level, graphs),
        viz.plot_cluster_vs_targets(school_level, graphs),
        viz.plot_feature_inventory(cfg, graphs),
        viz.plot_budget_ratio_correlation(school_level, cfg, graphs),
    ]
    for p in paths:
        print(f"    - {p.name:34s} ({p.stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 3 COMPLETE ✔  — school-year table with targets + engineered features.")
    print(_hr())


if __name__ == "__main__":
    main()
