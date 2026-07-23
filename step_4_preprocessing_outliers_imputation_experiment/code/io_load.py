"""
io_load.py — Step 4 input loading & shared feature prep.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Loads Step 3's school-level table and adds two derived columns reused across
Step 4 tasks:
  * combined_avg_grade  = mean(math_avg_grade, english_avg_grade)  (row-wise)
  * log_total_takers    = log1p(math_takers_total + english_takers_total)
``log_population`` is already computed in Step 3 (a deterministic transform is
a deterministic feature-engineering transform, not a preprocessing step).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    return (ROOT / rel_path).resolve()


def load_school_level(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return Step 3's school-level table with Step 4 derived columns added."""
    cfg = cfg or load_config()
    df = pd.read_csv(resolve(cfg["paths"]["school_level_in"]), encoding=cfg["io"]["encoding"])

    df["combined_avg_grade"] = df[["math_avg_grade", "english_avg_grade"]].mean(axis=1)
    total_takers = df[["math_takers_total", "english_takers_total"]].sum(axis=1, min_count=1)
    df["total_takers"] = total_takers
    df["log_total_takers"] = np.log1p(total_takers)
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = load_school_level(cfg)
    print(f"[io_load] school-level: {df.shape}")
    print("[io_load] derived cols added: combined_avg_grade, total_takers, log_total_takers")
