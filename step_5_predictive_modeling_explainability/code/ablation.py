"""
ablation.py — SES-only baseline vs Boruta-selected full feature set (NEW in v2).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

v1's Step 6 answered "does adding the budget dataset help?" as a bolted-on final
step. Having established that the budget data carries substantial independent
signal, we rebuilt the pipeline to merge all three datasets from the start
(Steps 1-2), so that framing no longer fits. The comparison itself, however, is
legitimate and valuable science — an ABLATION study — so it is kept here as a
Step-5 subsection: for each target we tune the champion HistGradientBoosting
TWICE on IDENTICAL rows — once on the original v1 feature set (SES only: cluster,
log_population, year, locality_form) and once on whatever Boruta selected from
the full SES+budget candidate space — so the R^2 delta is attributable ONLY to
the extra information, not to a different row set or protocol.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

import io_load as io
import modeling as ml


def build_ablation_xy(df: pd.DataFrame, target: str, cfg: dict[str, Any],
                      selected_numeric: list[str], selected_categorical: list[str]):
    """Build (X_before, X_after, y, groups) on IDENTICAL rows.

    The row mask requires the target AND every numeric column used by EITHER arm
    to be present, so both models are trained/evaluated on the same schools.
    """
    ab = cfg["ablation"]
    base_num, base_cat = ab["baseline_numeric"], ab["baseline_categorical"]
    group_col = cfg["features"]["group_col"]

    all_numeric = list(dict.fromkeys(base_num + selected_numeric))
    keep = df[target].notna()
    if all_numeric:
        keep = keep & df[all_numeric].notna().all(axis=1)

    sub = df[keep].reset_index(drop=True)
    X_before = io.encode_features(sub, base_num, base_cat)
    X_after = io.encode_features(sub, selected_numeric, selected_categorical)
    y = sub[target].reset_index(drop=True)
    groups = sub[group_col].reset_index(drop=True)
    return X_before, X_after, y, groups


def run_ablation_for_target(df: pd.DataFrame, target: str, cfg: dict[str, Any],
                            selected_numeric: list[str], selected_categorical: list[str]
                            ) -> dict[str, Any]:
    """Tune HGB on the SAME rows for the SES-only baseline and the full set."""
    X_before, X_after, y, groups = build_ablation_xy(
        df, target, cfg, selected_numeric, selected_categorical)

    before = ml.tune_champion(X_before, y, groups, cfg)
    after = ml.tune_champion(X_after, y, groups, cfg)

    bm, am = before["tuned_metrics"], after["tuned_metrics"]
    return {
        "target": target, "n_rows": len(y), "n_schools": int(groups.nunique()),
        "n_features_before": X_before.shape[1], "n_features_after": X_after.shape[1],
        "R2_before": bm["R2"], "R2_after": am["R2"], "dR2": am["R2"] - bm["R2"],
        "RMSE_before": bm["RMSE"], "RMSE_after": am["RMSE"],
        "MAE_before": bm["MAE"], "MAE_after": am["MAE"],
        "after_estimator": after["best_estimator"], "X_after": X_after, "y": y,
        "groups": groups,
    }
