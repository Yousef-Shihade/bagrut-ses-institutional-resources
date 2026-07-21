"""
run_step1.py — Step 1 orchestrator (v2: ingestion & standardisation, 3 datasets).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shehade & Shada Esawi

Flow:
    load + clean Bagrut / CBS / Budget  ->  cache 3 CSVs  ->  5 diagnostic plots
    ->  verification summary (shapes, keys, coverage, join feasibility)

Usage:
    python code/run_step1.py
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

import pandas as pd  # noqa: E402

import clean_text  # noqa: E402
import visualize as viz  # noqa: E402
from io_load import load_config, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 78


def main() -> None:
    cfg = load_config()
    graphs = resolve(cfg["paths"]["out_graphs"])

    print(_hr())
    print("STEP 1 — INGESTION & STANDARDIZATION (v2: THREE DATASETS)")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    # ---------------- load + clean ---------------------------------------- #
    res = clean_text.run(cfg)
    bag, ses, bud = res["bagrut"], res["ses"], res["budget"]
    ing, brep = res["budget_ingest"], res["budget_report"]

    print("\n[1] DATASET 1 — Bagrut grades")
    print(f"    shape {bag.shape} | schools(semel) {bag['semel'].nunique():,} | "
          f"cities {bag['city_norm'].nunique():,} | years {sorted(bag['year'].dropna().unique())}")
    print(f"    encoding utf-8-sig (BOM stripped); text columns trimmed; city_norm key built")

    print("\n[2] DATASET 2 — CBS socioeconomic index")
    print(f"    shape {ses.shape} | localities {ses['locality_norm'].nunique():,}")
    print(f"    cluster present {ses['cluster'].notna().mean()*100:.1f}% | "
          f"'..' placeholders -> NaN")

    print("\n[3] DATASET 3 — Ministry of Education budget  (NEW in v2)")
    print(f"    workbook: {ing['columns_in_workbook']} columns, {ing['rows_raw']:,} rows "
          f"(styles.xml colour error bypassed)")
    print(f"    totals rows dropped: {ing['totals_rows_dropped']} | "
          f"columns resolved: {ing['columns_resolved']}/{ing['columns_requested']}")
    if ing["columns_missing"]:
        print(f"    [!] NOT FOUND in workbook: {ing['columns_missing']}")
    print(f"    excluded up front (verified all-zero): {ing['known_empty_excluded']}")
    print(f"    clean shape {bud.shape} | unique semel {brep['unique_semel']:,} | "
          f"dup dropped {brep['duplicate_semel_dropped']}")
    print(f"    nurture quintile parsed for {brep['nurture_parsed_pct']}% of institutions")

    # ---------------- join feasibility (preview of Step 2) ----------------- #
    print("\n[4] JOIN FEASIBILITY (executed in Step 2)")
    bag_cities = set(bag["city_norm"].dropna())
    ses_cities = set(ses["locality_norm"].dropna())
    hit = len(bag_cities & ses_cities)
    rec_hit = bag["city_norm"].isin(ses_cities).mean() * 100
    print(f"    Bagrut <-> CBS   on normalised locality NAME:")
    print(f"        {hit}/{len(bag_cities)} distinct cities match exactly "
          f"({rec_hit:.1f}% of exam records) -> fuzzy stages close the rest in Step 2")

    bag_semels = set(pd.to_numeric(bag["semel"], errors="coerce").dropna().astype(int))
    bud_semels = set(bud["semel"])
    ov = len(bag_semels & bud_semels)
    print(f"    Bagrut <-> Budget on SCHOOL CODE (semel):")
    print(f"        {ov:,}/{len(bag_semels):,} schools match exactly "
          f"({100*ov/len(bag_semels):.1f}%) -> clean key join, no fuzzy needed")

    # ---------------- school-level attribute coverage ---------------------- #
    print("\n[5] NEW school-level attributes from Dataset 3")
    for col in ["district", "sector", "supervision", "legal_status",
                "education_stage", "nurture_quintile", "avg_class_size"]:
        if col in bud.columns:
            cov = bud[col].notna().mean() * 100
            nun = bud[col].nunique()
            print(f"    {col:20s} coverage {cov:5.1f}%   distinct {nun}")

    # ---------------- plots ------------------------------------------------ #
    print("\n[6] PLOTS")
    paths = [
        viz.plot_three_dataset_overview(bag, ses, bud, graphs),
        viz.plot_budget_column_coverage(bud, cfg, graphs),
        viz.plot_ses_cluster_frequency(ses, graphs),
        viz.plot_target_missingness(bag, cfg, graphs),
        viz.plot_budget_school_profile(bud, cfg, graphs),
    ]
    for p in paths:
        print(f"    - {p.name:34s} ({p.stat().st_size/1024:6.1f} KB)")

    # ---------------- outputs ---------------------------------------------- #
    print("\n[7] CACHED OUTPUTS")
    for k in ("bagrut", "ses", "budget"):
        p = res["paths"][k]
        print(f"    {p.name:22s} ({p.stat().st_size/1024/1024:5.2f} MB)")

    print("\n" + _hr())
    print("STEP 1 COMPLETE ✔  — three datasets ingested, standardised, cached.")
    print(_hr())


if __name__ == "__main__":
    main()
