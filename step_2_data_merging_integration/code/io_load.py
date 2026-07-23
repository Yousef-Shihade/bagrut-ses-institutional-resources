"""
io_load.py — Step 2 input loading (Step-1 clean caches, all three datasets).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    return (ROOT / rel_path).resolve()


def load_bagrut_clean(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["bagrut_clean"]), encoding=cfg["io"]["encoding"])


def load_ses_clean(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["ses_clean"]), encoding=cfg["io"]["encoding"])


def load_budget_clean(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["budget_clean"]), encoding=cfg["io"]["encoding"])
