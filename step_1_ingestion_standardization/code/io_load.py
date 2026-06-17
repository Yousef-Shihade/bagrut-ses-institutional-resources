"""
io_load.py — Resilient data ingestion for Step 1.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Responsibilities
----------------
* Locate the project root and load the central ``config.yaml``.
* Read Dataset 1 (Bagrut grades) handling its UTF-8 *BOM* so the first column
  header is not corrupted.
* Read Dataset 2 (CBS socioeconomic index) from an Excel sheet whose real header
  sits on row index 10, dropping the leading metadata/blank rows, renaming the
  Hebrew headers to canonical English names, and coercing the ``..`` "unranked"
  placeholders to ``NaN``.

These functions return raw (un-cleaned) DataFrames. Text normalisation lives in
``clean_text.py`` so that ingestion and cleaning stay independently testable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Step root = the directory that contains config.yaml (one level above code/).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load ``config.yaml`` as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return (ROOT / rel_path).resolve()


# --------------------------------------------------------------------------- #
# Dataset 1 — Bagrut grades
# --------------------------------------------------------------------------- #
def load_bagrut(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load ``israel_bagrut_averages.csv``.

    The file is encoded as UTF-8 *with BOM*. We read it with ``utf-8-sig`` and
    additionally guard against a stray BOM on the first header, so downstream
    code can rely on a clean ``grade`` column name.
    """
    cfg = cfg or load_config()
    path = resolve(cfg["paths"]["raw_bagrut"])
    encoding = cfg["bagrut"]["encoding"]

    df = pd.read_csv(path, encoding=encoding)

    # Belt-and-braces: drop any residual BOM and surrounding whitespace from the
    # column labels themselves.
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Dataset 2 — CBS socioeconomic index
# --------------------------------------------------------------------------- #
def _resolve_rename_map(raw_columns: list[str], rename_cfg: dict[str, str]) -> dict[str, str]:
    """Build a {raw_header -> canonical_name} map.

    Header labels in the workbook carry trailing spaces and, for the cluster
    column, an extra suffix ("אשכול    (מ-1 עד 10)"). We strip whitespace and
    match each configured Hebrew key by prefix so small formatting differences
    in the source file do not break ingestion.
    """
    mapping: dict[str, str] = {}
    for raw in raw_columns:
        key = str(raw).strip()
        for heb, canon in rename_cfg.items():
            if key == heb or key.startswith(heb):
                mapping[raw] = canon
                break
    return mapping


def load_ses(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the CBS socioeconomic index from the Excel workbook.

    Steps:
      1. Read sheet ``גיליון1`` using ``header=10`` (the real header row).
      2. Drop fully-empty rows (removes the blank row 11 and any trailing blanks).
      3. Rename Hebrew headers -> canonical English names.
      4. Coerce the ``..`` (and similar) unranked placeholders to ``NaN`` and cast
         the numeric columns to real numbers.
    """
    cfg = cfg or load_config()
    s = cfg["ses"]
    path = resolve(cfg["paths"]["raw_ses"])

    df = pd.read_excel(path, sheet_name=s["sheet"], header=s["header_row"])
    df = df.dropna(how="all").reset_index(drop=True)

    # Canonical English column names.
    rename_map = _resolve_rename_map(list(df.columns), s["rename"])
    df = df.rename(columns=rename_map)

    # Replace the unranked placeholder tokens with NaN, then make numerics numeric.
    df = df.replace({tok: pd.NA for tok in s["na_tokens"]})
    for col in s["numeric_columns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no locality code/name (defensive against stray footer rows).
    if "locality_name" in df.columns:
        df = df[df["locality_name"].notna()].reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Quick self-check when run directly.
    cfg = load_config()
    bag = load_bagrut(cfg)
    ses = load_ses(cfg)
    print(f"[io_load] bagrut: {bag.shape}  columns={list(bag.columns)}")
    print(f"[io_load] ses:    {ses.shape}  columns={list(ses.columns)}")
