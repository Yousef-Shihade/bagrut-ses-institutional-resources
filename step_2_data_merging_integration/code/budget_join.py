"""
budget_join.py — Join B: Bagrut <-> Budget on the school code (semel). NEW in v2.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shehade & Shada Esawi

Unlike the Bagrut<->CBS join (Join A, city-name fuzzy matching in matching.py),
``semel`` is a SCHOOL code in BOTH the Bagrut and Budget datasets, so this join
needs no fuzzy logic at all — just a clean key merge with type coercion and a
duplicate-key guard (the budget table is already unique per semel from Step 1,
but we verify defensively rather than assume).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def merge_budget(bagrut: pd.DataFrame, budget: pd.DataFrame,
                 cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-join budget features onto every Bagrut record via `semel`.

    Returns (merged_frame, diagnostics). All Bagrut rows are retained; schools
    absent from the budget file carry NaN in the budget columns.
    """
    bj = cfg["budget_join"]
    key = bj["key"]
    feat_cols = [c for c in bj["feature_columns"] if c in budget.columns]

    out = bagrut.copy()
    out[key] = pd.to_numeric(out[key], errors="coerce")

    bud = budget.copy()
    bud[key] = pd.to_numeric(bud[key], errors="coerce")
    dup = bud[key].duplicated().sum()
    if dup:
        bud = bud.drop_duplicates(subset=[key], keep="first")

    bud_feats = bud[[key] + feat_cols]
    out = out.merge(bud_feats, on=key, how="left", suffixes=("", "_budget"))

    matched_rows = int(out[feat_cols[0]].notna().sum()) if feat_cols else 0
    matched_schools = int(out.loc[out[feat_cols[0]].notna(), key].nunique()) if feat_cols else 0
    total_schools = int(out[key].nunique())

    diag = {
        "budget_duplicate_semel_dropped": int(dup),
        "rows_total": int(len(out)),
        "rows_with_budget": matched_rows,
        "rows_with_budget_pct": round(100 * matched_rows / len(out), 2),
        "schools_total": total_schools,
        "schools_matched": matched_schools,
        "schools_matched_pct": round(100 * matched_schools / total_schools, 2),
    }
    return out, diag
