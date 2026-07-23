"""
run_step2.py — Step 2 orchestrator (three-way merge).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Flow:
    load 3 Step-1 caches
    -> Join A: Bagrut <-> CBS  (4-stage fuzzy locality-name matching)
    -> Join B: Bagrut <-> Budget (exact semel key)
    -> one consolidated record-level table (still 69,638 rows — left joins only)
    -> 3 diagnostic plots + verification summary

Usage:
    python code/run_step2.py
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

import budget_join as bj  # noqa: E402
import matching  # noqa: E402
import visualize as viz  # noqa: E402
from io_load import load_budget_clean, load_bagrut_clean, load_config, load_ses_clean, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 78


def main() -> None:
    cfg = load_config()
    graphs = resolve(cfg["paths"]["out_graphs"])
    data_dir = resolve(cfg["paths"]["out_data"])
    data_dir.mkdir(parents=True, exist_ok=True)

    print(_hr())
    print("STEP 2 — THREE-WAY MERGE")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    bagrut = load_bagrut_clean(cfg)
    ses = load_ses_clean(cfg)
    budget = load_budget_clean(cfg)
    print(f"\n[LOAD] bagrut {bagrut.shape} | ses {ses.shape} | budget {budget.shape}")

    # ---------------- Join A: Bagrut <-> CBS (fuzzy locality name) ---------- #
    m = cfg["matching"]
    ses_dedup = matching.dedup_ses(ses, key=m["ses_key"], by=m["ses_dedup_by"])
    mapping = matching.build_city_mapping(bagrut, ses_dedup, ses, cfg)
    merged = matching.merge(bagrut, ses, mapping, cfg)

    stage_counts = mapping.groupby("stage")["n_records"].sum()
    total = stage_counts.sum()
    print("\n[JOIN A] Bagrut <-> CBS  (locality name, 4-stage fuzzy)")
    for stage in ["exact", "structural", "crosswalk", "fuzzy", "unmatched"]:
        n = int(stage_counts.get(stage, 0))
        print(f"    {stage:12s} {n:7,d} records  ({100*n/total:5.2f}%)")
    matched_pct_a = 100 * (total - stage_counts.get("unmatched", 0)) / total
    print(f"    TOTAL MATCHED: {matched_pct_a:.2f}%  |  {mapping['stage'].eq('unmatched').sum()} "
          f"distinct cities unmatched")

    # ---------------- Join B: Bagrut <-> Budget (exact semel) --------------- #
    merged, diag_b = bj.merge_budget(merged, budget, cfg)
    print("\n[JOIN B] Bagrut <-> Budget  (school code `semel`, exact key)")
    print(f"    schools matched: {diag_b['schools_matched']:,}/{diag_b['schools_total']:,} "
          f"({diag_b['schools_matched_pct']}%)")
    print(f"    rows matched:    {diag_b['rows_with_budget']:,}/{diag_b['rows_total']:,} "
          f"({diag_b['rows_with_budget_pct']}%)")
    if diag_b["budget_duplicate_semel_dropped"]:
        print(f"    [!] {diag_b['budget_duplicate_semel_dropped']} duplicate semel rows "
              f"dropped from the budget table before merging")

    # ---------------- consolidated matrix ------------------------------------ #
    print("\n[CONSOLIDATED MATRIX]")
    print(f"    shape: {merged.shape[0]:,} rows x {merged.shape[1]} columns")
    print(f"    (row count unchanged from Bagrut input — both joins are LEFT joins,")
    print(f"     no row explosion; unmatched cells carry NaN, filtered later)")
    n_cols_ses = len([c for c in cfg["ses_feature_columns"] if c in merged.columns])
    n_cols_bud = len([c for c in cfg["budget_join"]["feature_columns"] if c in merged.columns])
    print(f"    CBS columns attached:    {n_cols_ses}")
    print(f"    Budget columns attached: {n_cols_bud}")

    # both-matched coverage (rows with a full profile from all 3 sources)
    both = merged["cluster"].notna() & merged[cfg["budget_join"]["feature_columns"][0]].notna()
    print(f"    rows with BOTH CBS + Budget attached: {both.sum():,} "
          f"({100*both.mean():.2f}%) — the fully-consolidated core")

    # ---------------- save ---------------------------------------------------- #
    mapping.to_csv(resolve(cfg["paths"]["mapping_out"]), index=False, encoding=cfg["io"]["encoding"])
    merged.to_csv(resolve(cfg["paths"]["merged_out"]), index=False, encoding=cfg["io"]["encoding"])
    print(f"\n[SAVE] {Path(cfg['paths']['mapping_out']).name} "
          f"({resolve(cfg['paths']['mapping_out']).stat().st_size/1024:.1f} KB)")
    print(f"[SAVE] {Path(cfg['paths']['merged_out']).name} "
          f"({resolve(cfg['paths']['merged_out']).stat().st_size/1024/1024:.2f} MB)")

    # ---------------- plots ---------------------------------------------------- #
    print("\n[PLOTS]")
    paths = [
        viz.plot_match_yield_waterfall(mapping, graphs),
        viz.plot_dual_join_success(matched_pct_a, diag_b["rows_with_budget_pct"],
                                   "Join A: Bagrut ↔ CBS\n(fuzzy, locality name)",
                                   "Join B: Bagrut ↔ Budget\n(exact, semel)", graphs),
        viz.plot_sector_supervision_by_cluster(merged, cfg, graphs),
    ]
    for p in paths:
        print(f"    - {p.name:34s} ({p.stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 2 COMPLETE ✔  — three datasets consolidated into one record-level table.")
    print(_hr())


if __name__ == "__main__":
    main()
