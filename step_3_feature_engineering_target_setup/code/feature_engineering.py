"""
feature_engineering.py — re-grain to school level + engineer targets & features.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Transforms the Step 2 three-way-merged record table into a SCHOOL-level
(semel x year) table with:

  * 4 targets — Math/English x {takers-weighted avg grade, 5-unit participation}
    (``_aggregate_subject`` is unchanged from v1: it is dataset-agnostic and was
    already validated).

  * Municipal predictors (CBS) — cluster, index_value, population, locality_form.

  * NEW school-level predictors (budget dataset) — 5 categoricals (district,
    sector, supervision, legal_status, education_stage), 2 direct numerics
    (nurture_quintile, avg_class_size), and 8 ENGINEERED per-student ratios
    (total/teaching/tuition/perimeter/projects/purchases/transport budget per
    student, private hours per student) plus special_ed_share and
    log_school_size.

We never impute a target: a grade target is NaN only when every cell for that
subject/school/year was suppressed (privacy censoring, see Step 1); a
participation target is NaN only when the school had no academic-track takers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Targets — unchanged from v1 (dataset-agnostic, already validated at 99.44%   #
# merge quality; only the input table's provenance changed).                   #
# --------------------------------------------------------------------------- #
def _aggregate_subject(
    df: pd.DataFrame,
    subject: str,
    prefix: str,
    grain: list[str],
    part_units: list[int],
    adv_units: int,
) -> pd.DataFrame:
    """Vectorised per-(grain) aggregation for one subject.

    Returns columns: <prefix>_avg_grade, <prefix>_5unit_participation,
    <prefix>_takers_total, indexed by ``grain``.
    """
    s = df[df["subject"] == subject].copy()
    if s.empty:
        return pd.DataFrame(
            columns=[f"{prefix}_avg_grade", f"{prefix}_5unit_participation",
                     f"{prefix}_takers_total"]
        )

    s["_grade_x_takers"] = s["grade"] * s["takers"]
    s["_takers_observed"] = s["takers"].where(s["grade"].notna(), 0)
    s["_takers_adv"] = s["takers"].where(s["studyunits"] == adv_units, 0)
    s["_takers_track"] = s["takers"].where(s["studyunits"].isin(part_units), 0)

    agg = s.groupby(grain).agg(
        _gxt=("_grade_x_takers", "sum"),
        _t_obs=("_takers_observed", "sum"),
        _t_adv=("_takers_adv", "sum"),
        _t_track=("_takers_track", "sum"),
        _t_total=("takers", "sum"),
    )

    avg_grade = np.where(agg["_t_obs"] > 0, agg["_gxt"] / agg["_t_obs"].replace(0, np.nan),
                         np.nan)
    participation = np.where(agg["_t_track"] > 0,
                             agg["_t_adv"] / agg["_t_track"].replace(0, np.nan), np.nan)

    out = pd.DataFrame({
        f"{prefix}_avg_grade": avg_grade,
        f"{prefix}_5unit_participation": participation,
        f"{prefix}_takers_total": agg["_t_total"].astype(int),
    }, index=agg.index)
    return out


# --------------------------------------------------------------------------- #
# NEW — budget ratio engineering                                               #
# --------------------------------------------------------------------------- #
def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator with inf and non-positive-denominator -> NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = numerator / denominator
    r = r.replace([np.inf, -np.inf], np.nan)
    r[denominator.fillna(0) <= 0] = np.nan
    return r


def engineer_budget_features(school_level: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add the 8 per-student ratios + special_ed_share + log_school_size.

    Operates on the already school-year-graded table (budget columns are
    constant per semel, so this is safe to run post-aggregation).
    """
    out = school_level.copy()

    for name, (num_col, den_col) in cfg["budget_ratios"].items():
        if num_col in out.columns and den_col in out.columns:
            out[name] = _safe_ratio(out[num_col], out[den_col])

    if {"students_regular", "students_special"}.issubset(out.columns):
        total = out["students_regular"].fillna(0) + out["students_special"].fillna(0)
        out["special_ed_share"] = _safe_ratio(out["students_special"], total)
        out["log_school_size"] = np.log1p(total.where(total > 0))

    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def build_school_level(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the school-level (semel x year) feature/target table."""
    grain = cfg["grain"]
    core = cfg["subjects"]["core"]
    part_units = cfg["subjects"]["participation_units"]
    adv_units = cfg["subjects"]["advanced_units"]

    math = _aggregate_subject(df, core["math"], "math", grain, part_units, adv_units)
    eng = _aggregate_subject(df, core["english"], "english", grain, part_units, adv_units)
    targets = math.join(eng, how="outer")

    # Static metadata per grain: CBS + budget categoricals/direct + ratio
    # ingredients (constant within a semel — Step 1/2 broadcast them).
    meta_cols = (cfg["cbs_features"] + cfg["budget_categorical"]
                + cfg["budget_direct_numeric"] + cfg["budget_ratio_ingredients"]
                + cfg["id_columns"])
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df.groupby(grain)[meta_cols].first()

    school_level = meta.join(targets, how="right").reset_index()
    school_level = engineer_budget_features(school_level, cfg)

    # log_population — a standard transform for a heavy-tailed size variable;
    # done here (feature engineering) rather than in a later preprocessing step,
    # since it is deterministic and has no leakage risk.
    if "population" in school_level.columns:
        school_level["log_population"] = np.log1p(school_level["population"])

    target_cols = ["math_avg_grade", "english_avg_grade",
                   "math_5unit_participation", "english_5unit_participation"]
    support_cols = ["math_takers_total", "english_takers_total"]
    ratio_cols = list(cfg["budget_ratios"].keys()) + ["special_ed_share", "log_school_size"]

    ordered = (grain + cfg["id_columns"] + cfg["cbs_features"] + ["log_population"]
              + cfg["budget_categorical"] + cfg["budget_direct_numeric"] + ratio_cols
              + target_cols + support_cols + cfg["budget_ratio_ingredients"])
    ordered = list(dict.fromkeys(c for c in ordered if c in school_level.columns))
    school_level = school_level[ordered]
    return school_level.sort_values(grain).reset_index(drop=True)
