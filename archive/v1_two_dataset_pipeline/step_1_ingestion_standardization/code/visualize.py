"""
visualize.py — Step 1 exploratory plots.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Generates four diagnostic figures (saved as PNG into ``outputs/graphs``) that we
inspect *before* merging the datasets:

  1. raw_grade_distribution.png   — histogram of the observed Bagrut ``grade``.
  2. studyunits_breakdown.png     — record counts per study-unit level (2/3/4/5).
  3. ses_cluster_frequency.png    — locality counts per socioeconomic cluster 1-10.
  4. target_missingness.png       — ``takers`` distribution, grade-missing vs
                                    grade-present (validates the small-cohort
                                    privacy-suppression hypothesis).

All labels are kept in English on purpose: matplotlib's default fonts do not
shape right-to-left Hebrew well, and none of these four plots needs Hebrew text.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / file-only backend

# seaborn 0.12 emits FutureWarnings for the palette/use_inf_as_na paths we rely
# on; silence them so the Step-1 console output stays clean for the report.
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from io_load import load_config, resolve

sns.set_theme(style="whitegrid", context="talk")
_PALETTE = "viridis"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grade_distribution(bagrut: pd.DataFrame, out_dir: Path, cfg: dict) -> Path:
    """Plot 1 — distribution of the observed grade column."""
    grade = bagrut[cfg["bagrut"]["grade_column"]].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(grade, bins=40, kde=True, color="#2a788e", ax=ax)
    ax.axvline(grade.mean(), color="crimson", ls="--", lw=2,
               label=f"mean = {grade.mean():.1f}")
    ax.axvline(grade.median(), color="darkorange", ls=":", lw=2,
               label=f"median = {grade.median():.1f}")
    ax.set_title("Plot 1 — Observed Bagrut Grade Distribution")
    ax.set_xlabel("Average Bagrut grade (subject-cell)")
    ax.set_ylabel("Count")
    ax.legend()
    return _save(fig, out_dir, "raw_grade_distribution.png")


def plot_studyunits_breakdown(bagrut: pd.DataFrame, out_dir: Path, cfg: dict) -> Path:
    """Plot 2 — record counts per study-unit level."""
    col = cfg["bagrut"]["studyunits_column"]
    counts = bagrut[col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = sns.barplot(x=counts.index.astype(str), y=counts.values,
                       palette=_PALETTE, ax=ax)
    for p, v in zip(bars.patches, counts.values):
        ax.annotate(f"{v:,}", (p.get_x() + p.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=12)
    adv = cfg["subjects"]["advanced_units"]
    ax.set_title(f"Plot 2 — Records by Study-Unit Level ({adv}u = advanced track)")
    ax.set_xlabel("Study units")
    ax.set_ylabel("Number of records")
    ax.margins(y=0.12)
    return _save(fig, out_dir, "studyunits_breakdown.png")


def plot_cluster_frequency(ses: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 3 — locality counts per socioeconomic cluster (1-10)."""
    clusters = ses["cluster"].dropna().astype(int)
    counts = clusters.value_counts().reindex(range(1, 11), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=list(counts.index), y=counts.values,
                       palette="rocket", ax=ax)
    for p, v in zip(bars.patches, counts.values):
        ax.annotate(f"{v}", (p.get_x() + p.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11)
    n_unranked = int(ses["cluster"].isna().sum())
    ax.set_title("Plot 3 — Localities per Socioeconomic Cluster (skewed to 6-8)")
    ax.set_xlabel("CBS socioeconomic cluster (1 = lowest, 10 = highest)")
    ax.set_ylabel("Number of localities")
    ax.margins(y=0.12)
    ax.text(0.99, 0.95, f"unranked ('..'): {n_unranked}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#cc9a06"))
    return _save(fig, out_dir, "ses_cluster_frequency.png")


def plot_target_missingness(bagrut: pd.DataFrame, out_dir: Path, cfg: dict) -> Path:
    """Plot 4 — cohort size (takers) vs grade missingness.

    The privacy-suppression hypothesis predicts that *missing* grades belong to
    much smaller cohorts. A log-scaled boxplot makes the gap unmistakable.
    """
    grade_col = cfg["bagrut"]["grade_column"]
    takers_col = cfg["bagrut"]["takers_column"]
    df = bagrut[[grade_col, takers_col]].copy()
    df["grade_status"] = np.where(df[grade_col].isna(), "Missing grade", "Present grade")

    med = df.groupby("grade_status")[takers_col].median()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="grade_status", y=takers_col,
                order=["Missing grade", "Present grade"],
                palette={"Missing grade": "#d1495b", "Present grade": "#30638e"},
                ax=ax)
    ax.set_yscale("log")
    ax.set_title("Plot 4 — Cohort Size vs Grade Missingness\n"
                 "(validates small-cohort privacy suppression)")
    ax.set_xlabel("")
    ax.set_ylabel("Test-takers per cell (log scale)")
    for i, status in enumerate(["Missing grade", "Present grade"]):
        if status in med.index:
            ax.annotate(f"median = {med[status]:.0f}", (i, med[status]),
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
    return _save(fig, out_dir, "target_missingness.png")


def run(bagrut: pd.DataFrame, ses: pd.DataFrame,
        cfg: dict[str, Any] | None = None) -> list[Path]:
    """Generate all four Step-1 plots; return their paths."""
    cfg = cfg or load_config()
    out_dir = resolve(cfg["paths"]["out_graphs"])
    paths = [
        plot_grade_distribution(bagrut, out_dir, cfg),
        plot_studyunits_breakdown(bagrut, out_dir, cfg),
        plot_cluster_frequency(ses, out_dir),
        plot_target_missingness(bagrut, out_dir, cfg),
    ]
    return paths


if __name__ == "__main__":
    import clean_text

    cfg = load_config()
    res = clean_text.run(cfg)
    for p in run(res["bagrut"], res["ses"], cfg):
        print(f"[visualize] saved {p}")
