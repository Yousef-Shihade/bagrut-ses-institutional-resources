"""
run_step1.py — Step 1 orchestrator (Ingestion & Text Standardisation).

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Runs the full Step-1 pipeline end to end and prints a verification summary:

    load -> clean (cache CSVs) -> visualise (4 PNGs) -> report

Usage (from anywhere):
    python src/run_step1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling modules importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import clean_text  # noqa: E402
import visualize  # noqa: E402
from io_load import ROOT, load_bagrut, load_config, resolve  # noqa: E402


def _hr(char: str = "=") -> str:
    return char * 70


def main() -> None:
    cfg = load_config()

    print(_hr())
    print("STEP 1 — INGESTION & TEXT STANDARDISATION")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    # --- 1. Load + clean (caches outputs/data/*.csv) ----------------------- #
    raw_bagrut = load_bagrut(cfg)
    res = clean_text.run(cfg)
    bag, ses = res["bagrut"], res["ses"]

    # --- 2. Visualise ------------------------------------------------------ #
    plot_paths = visualize.run(bag, ses, cfg)

    # --- 3. Verification summary ------------------------------------------ #
    grade_col = cfg["bagrut"]["grade_column"]
    takers_col = cfg["bagrut"]["takers_column"]
    miss = raw_bagrut[grade_col].isna()

    print("\n[1] DATA SHAPES")
    print(f"    Bagrut (raw/clean) : {raw_bagrut.shape} -> {bag.shape}")
    print(f"    SES    (clean)     : {ses.shape}")
    print(f"    Bagrut columns     : {list(bag.columns)}")
    print(f"    SES    columns     : {list(ses.columns)}")

    print("\n[2] TARGET MISSINGNESS (grade)")
    print(f"    missing            : {miss.sum():,} / {len(raw_bagrut):,} "
          f"({miss.mean()*100:.1f}%)")
    print(f"    median takers | missing grade : {raw_bagrut.loc[miss, takers_col].median():.0f}")
    print(f"    median takers | present grade : {raw_bagrut.loc[~miss, takers_col].median():.0f}")

    print("\n[3] TEXT STANDARDISATION CHECK")
    n_pad_before = raw_bagrut["city"].astype(str).str.endswith(" ").sum()
    n_pad_after = bag["city_norm"].astype(str).str.endswith(" ").sum()
    print(f"    Bagrut cities w/ trailing space : {n_pad_before:,} -> {n_pad_after:,}")
    print(f"    unique city_norm keys (bagrut)  : {bag['city_norm'].nunique():,}")
    print(f"    unique locality_norm keys (ses) : {ses['locality_norm'].nunique():,}")
    overlap = set(bag["city_norm"].dropna()) & set(ses["locality_norm"].dropna())
    cov = bag["city_norm"].isin(overlap).mean() * 100
    print(f"    normalised exact-key overlap    : {len(overlap)} localities "
          f"({cov:.1f}% of bagrut records)  [full matching is Step 2]")

    print("\n[4] CLUSTER SUMMARY (ses)")
    print(f"    ranked localities  : {int(ses['cluster'].notna().sum())}")
    print(f"    unranked ('..')    : {int(ses['cluster'].isna().sum())}")

    print("\n[5] ARTEFACTS WRITTEN")
    print(f"    interim data dir   : {resolve(cfg['paths']['out_data'])}")
    print(f"      - {Path(res['bagrut_path']).name}")
    print(f"      - {Path(res['ses_path']).name}")
    print(f"    graphs dir         : {resolve(cfg['paths']['out_graphs'])}")
    for p in plot_paths:
        size_kb = Path(p).stat().st_size / 1024
        print(f"      - {Path(p).name:32s} ({size_kb:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 1 COMPLETE ✔   (awaiting signal for Step 2: Merging)")
    print(_hr())


if __name__ == "__main__":
    main()
