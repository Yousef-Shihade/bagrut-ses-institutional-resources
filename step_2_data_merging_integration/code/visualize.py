"""
visualize.py — Step 2 diagnostic plots (three-way merge, v2).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shehade & Shada Esawi

Plots:
1. match_yield_waterfall.png  — Join A (CBS) stage-by-stage yield (exact ->
                                structural -> crosswalk -> fuzzy -> unmatched).
2. dual_join_success.png      — Join A vs Join B match rate, side by side (the
                                new v2 story: one join is fuzzy/hard, one is
                                clean/easy, and BOTH feed the same final table).
3. sector_supervision_by_cluster.png — NEW: does the budget dataset's school-level
                                sector/supervision correlate with municipal SES
                                cluster, or is it an independent axis of variation?
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk")

NAVY, TEAL, CORAL, GOLD = "#1b2a4a", "#2a9d8f", "#d1495b", "#e8b23a"
STAGE_COLORS = {"exact": TEAL, "structural": "#5aa6a0", "crosswalk": GOLD,
                "fuzzy": "#c98a3a", "unmatched": CORAL}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_match_yield_waterfall(mapping: pd.DataFrame, out_dir: Path) -> Path:
    """Join A (Bagrut<->CBS): records matched at each stage, in order."""
    order = ["exact", "structural", "crosswalk", "fuzzy", "unmatched"]
    by_stage = mapping.groupby("stage")["n_records"].sum().reindex(order, fill_value=0)
    total = by_stage.sum()
    pct = 100 * by_stage / total

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(order, by_stage.values, color=[STAGE_COLORS[s] for s in order])
    for b, v, p in zip(bars, by_stage.values, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({p:.2f}%)",
                ha="center", va="bottom", fontsize=11)
    cum_matched = 100 - pct["unmatched"]
    ax.set_title(f"Join A — Bagrut ↔ CBS locality-name match yield\n"
                 f"cumulative matched = {cum_matched:.2f}% of exam records", fontsize=15)
    ax.set_ylabel("Bagrut exam records")
    return _save(fig, out_dir, "match_yield_waterfall.png")


def plot_dual_join_success(join_a_pct: float, join_b_pct: float,
                           join_a_label: str, join_b_label: str,
                           out_dir: Path) -> Path:
    """Side-by-side match rate for the two independent joins feeding Step 2."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [join_a_label, join_b_label]
    vals = [join_a_pct, join_b_pct]
    colors = [TEAL, GOLD]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center",
                va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.axhline(100, color="grey", ls="--", lw=1)
    ax.set_ylabel("% of Bagrut exam records matched")
    ax.set_title("Two independent joins feed one consolidated table", fontsize=15)
    return _save(fig, out_dir, "dual_join_success.png")


def plot_sector_supervision_by_cluster(merged: pd.DataFrame, cfg: dict[str, Any],
                                       out_dir: Path) -> Path:
    """Does the school-level sector/supervision track the municipal cluster?

    If NOT (i.e. sector/supervision varies widely within every cluster), that is
    evidence the budget dataset adds an INDEPENDENT axis of variation, not a proxy
    for municipal SES that Boruta would find redundant.
    """
    labels = cfg.get("display_labels", {})
    d = merged.dropna(subset=["cluster", "sector"]).copy()
    d["cluster"] = d["cluster"].astype(int)
    d["sector_en"] = d["sector"].map(lambda v: labels.get("sector", {}).get(v, v))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ct = pd.crosstab(d["cluster"], d["sector_en"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax,
           color=sns.color_palette("Set2", n_colors=ct.shape[1]))
    ax.set_title("School sector composition WITHIN each socioeconomic cluster\n"
                 "(if bars vary a lot by cluster, sector is not a cluster proxy)",
                 fontsize=14)
    ax.set_xlabel("CBS socioeconomic cluster")
    ax.set_ylabel("% of exam records")
    ax.legend(title="Sector", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    return _save(fig, out_dir, "sector_supervision_by_cluster.png")
