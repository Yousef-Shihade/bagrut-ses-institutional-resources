"""
clean_text.py — Hebrew string standardisation for Step 1.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Why this module exists
----------------------
The two datasets must eventually be joined on *locality name* (the ``semel``
codes are NOT compatible: in the Bagrut file ``semel`` is a *school* code, while
in the CBS file it is a *locality* code — zero overlap). Locality names, however,
differ between the sources because of:

  * trailing / leading whitespace padding (every Bagrut text value),
  * collapsed multiple internal spaces,
  * parenthetical qualifiers such as (יישוב) / (קבוצה) / (איחוד) / (מוסד),
  * Hebrew spelling variants — most importantly the "yod-doubling" convention
    (קרית -> קריית, הרצליה -> הרצלייה, נהריה -> נהרייה ...).

This module produces a normalised key column (``city_norm`` / ``locality_norm``)
on each dataset while preserving the original text. The actual fuzzy/crosswalk
matching happens in Step 2; here we only standardise the strings and cache the
cleaned tables.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from io_load import load_bagrut, load_ses, load_config, resolve

# Gershayim / geresh / quotation marks that show up inside Hebrew names.
_QUOTES = "\"'״׳‘’“”`"
_PAREN_RE = re.compile(r"\([^)]*\)")          # (...) including the brackets
_LEFTOVER_PAREN_RE = re.compile(r"[()\[\]{}]")
_WS_RE = re.compile(r"\s+")


def normalize_hebrew(
    value: Any,
    *,
    strip_parentheses: bool = True,
    yod_prefix_fixes: dict[str, str] | None = None,
    spelling_map: dict[str, str] | None = None,
) -> str | float:
    """Return a normalised locality string (or ``pd.NA`` for missing input).

    Order of operations matters: structural cleaning (whitespace, parentheses,
    quotes) happens first so the spelling rules see a canonical string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA

    text = str(value)

    # 1. Remove parenthetical qualifiers, then any stray bracket characters.
    if strip_parentheses:
        text = _PAREN_RE.sub(" ", text)
        text = _LEFTOVER_PAREN_RE.sub(" ", text)

    # 2. Drop Hebrew/ASCII quote marks.
    text = text.translate({ord(ch): " " for ch in _QUOTES})

    # 3. Collapse whitespace and trim.
    text = _WS_RE.sub(" ", text).strip()

    # 4. Generic yod-prefix fixes (word-initial only, e.g. קרית -> קריית).
    for src, dst in (yod_prefix_fixes or {}).items():
        text = re.sub(rf"(^|\s){re.escape(src)}(?=\s|$)", rf"\1{dst}", text)

    # 5. Documented whole-string spelling variants (Bagrut -> CBS spelling).
    if spelling_map and text in spelling_map:
        text = spelling_map[text]

    return text if text else pd.NA


def _rules_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg.get("text_cleaning", {})
    return {
        "strip_parentheses": tc.get("strip_parentheses", True),
        "yod_prefix_fixes": tc.get("yod_prefix_fixes", {}) or {},
        "spelling_map": tc.get("spelling_map", {}) or {},
    }


def clean_string_column(s: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Vectorised application of :func:`normalize_hebrew` to a Series."""
    rules = _rules_from_cfg(cfg)
    return s.map(lambda v: normalize_hebrew(v, **rules))


def strip_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Light strip+collapse on listed text columns (keeps original semantics).

    Used for the Bagrut ``subject`` / ``school`` columns, which only need the
    trailing-space padding removed (no spelling normalisation).
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = (
                out[col].astype("string").str.replace(_WS_RE, " ", regex=True).str.strip()
            )
    return out


# --------------------------------------------------------------------------- #
# Dataset-level cleaning
# --------------------------------------------------------------------------- #
def clean_bagrut(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Strip the padded text columns and add a normalised ``city_norm`` key."""
    text_cols = cfg["bagrut"]["text_columns"]
    city_col = cfg["bagrut"]["city_column"]

    out = strip_text_columns(df, text_cols)
    out["city_norm"] = clean_string_column(out[city_col], cfg)
    return out


def clean_ses(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add a normalised ``locality_norm`` key to the CBS table."""
    name_col = cfg["ses"]["name_column"]
    out = df.copy()
    out[name_col] = out[name_col].astype("string").str.replace(_WS_RE, " ", regex=True).str.strip()
    out["locality_norm"] = clean_string_column(out[name_col], cfg)
    return out


def run(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Load -> clean both datasets and cache them to ``outputs/data``.

    Returns the cleaned DataFrames so callers (e.g. the visualiser / orchestrator)
    can reuse them without re-reading from disk.
    """
    cfg = cfg or load_config()

    bag_clean = clean_bagrut(load_bagrut(cfg), cfg)
    ses_clean = clean_ses(load_ses(cfg), cfg)

    out_dir = resolve(cfg["paths"]["out_data"])
    out_dir.mkdir(parents=True, exist_ok=True)
    bag_path = out_dir / "bagrut_clean.csv"
    ses_path = out_dir / "ses_clean.csv"

    # utf-8-sig so the cached CSVs open cleanly in Excel with Hebrew intact.
    bag_clean.to_csv(bag_path, index=False, encoding="utf-8-sig")
    ses_clean.to_csv(ses_path, index=False, encoding="utf-8-sig")

    return {
        "bagrut": bag_clean,
        "ses": ses_clean,
        "bagrut_path": bag_path,
        "ses_path": ses_path,
    }


if __name__ == "__main__":
    cfg = load_config()
    res = run(cfg)
    bag, ses = res["bagrut"], res["ses"]
    print(f"[clean_text] bagrut clean: {bag.shape} -> {res['bagrut_path']}")
    print(f"[clean_text] ses    clean: {ses.shape} -> {res['ses_path']}")
    # Show a few before/after examples of the normalisation.
    raw_bag = load_bagrut(cfg)
    sample = (
        pd.DataFrame({"raw": raw_bag["city"], "norm": bag["city_norm"]})
        .drop_duplicates()
        .head(6)
    )
    print("[clean_text] sample city normalisation:")
    for _, r in sample.iterrows():
        print(f"    {r['raw']!r:35s} -> {r['norm']!r}")
