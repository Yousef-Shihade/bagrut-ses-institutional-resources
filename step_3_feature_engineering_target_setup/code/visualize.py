"""
visualize.py — Step 3 diagnostic plots (v2: targets + engineered features).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk")

NAVY, TEAL, CORAL, GOLD = "#1b2a4a", "#2a9d8f", "#d1495b", "#e8b23a"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distributions(school_level: pd.DataFrame, out_dir: Path) -> Path:
    targets = ["math_avg_grade", "english_avg_grade",
              "math_5unit_participation", "english_5unit_participation"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, t in zip(axes.flat, targets):
        sns.histplot(school_level[t].dropna(), bins=40, kde=True,
                    color=TEAL if "math" in t else NAVY, ax=ax)
        ax.set_title(t)
    fig.suptitle("Step 3 — Target distributions (school x year)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_dir, "target_distributions.png")


def plot_cluster_vs_targets(school_level: pd.DataFrame, out_dir: Path) -> Path:
    d = school_level.dropna(subset=["cluster"]).copy()
    d["cluster"] = d["cluster"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=d, x="cluster", y="math_avg_grade", color=TEAL,
               showfliers=False, ax=axes[0])
    axes[0].set_title("Math avg grade by cluster")
    sns.boxplot(data=d, x="cluster", y="english_avg_grade", color=NAVY,
               showfliers=False, ax=axes[1])
    axes[1].set_title("English avg grade by cluster")
    fig.suptitle("Cluster vs Bagrut outcomes (municipal SES gradient)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir, "cluster_vs_targets.png")


def plot_feature_inventory(cfg: dict[str, Any], out_dir: Path) -> Path:
    """Bar chart of the final feature count by source — quantifies how far the
    third dataset widened the previously municipal-only feature space."""
    counts = {
        "CBS\n(municipal)": len(cfg["cbs_features"]) - 1,  # exclude ses_locality_name (id, not a feature)
        "Budget\ncategorical": len(cfg["budget_categorical"]),
        "Budget\nnumeric": len(cfg["budget_direct_numeric"]) + len(cfg["budget_ratios"]) + 2,  # +special_ed_share +log_school_size
        "Temporal": 1,  # year
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(list(counts), list(counts.values()),
                  color=[TEAL, GOLD, "#c98a3a", NAVY])
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("candidate features")
    total = sum(counts.values())
    ax.set_title(f"Step 3 — Candidate feature inventory  (total = {total})", fontsize=15)
    return _save(fig, out_dir, "feature_inventory.png")


def plot_budget_ratio_correlation(school_level: pd.DataFrame, cfg: dict[str, Any],
                                  out_dir: Path) -> Path:
    """Correlation of the 8 engineered budget ratios with the municipal cluster —
    tests whether they are new/orthogonal information or a cluster proxy."""
    ratio_cols = [c for c in cfg["budget_ratios"].keys() if c in school_level.columns]
    cols = ["cluster"] + ratio_cols + ["avg_class_size", "special_ed_share"]
    cols = [c for c in cols if c in school_level.columns]
    corr = school_level[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
               cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Budget-derived features vs municipal cluster\n"
                 "(near-zero = independent new information)", fontsize=14)
    return _save(fig, out_dir, "budget_ratio_correlation.png")
