"""io_load.py — Step 3 input loading.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shehade & Shada Esawi
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


def load_merged(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["merged_in"]), encoding=cfg["io"]["encoding"])
