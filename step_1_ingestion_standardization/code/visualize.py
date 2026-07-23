"""
visualize.py — Step 1 diagnostic plots (three datasets).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Plots produced
--------------
1. three_dataset_overview.png   — scale & key structure of all three sources.
2. budget_column_coverage.png   — which budget columns are actually usable
                                  (the audit that exposed the all-zero Gefen and
                                  guidance-hours columns).
3. ses_cluster_frequency.png    — distribution of the CBS socioeconomic cluster.
4. bagrut_target_missingness.png— the 21% grade suppression (missing NOT at random).
5. budget_school_profile.png    — NEW school-level categoricals (sector,
                                  supervision, district, nurture quintile).
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


def plot_three_dataset_overview(bag: pd.DataFrame, ses: pd.DataFrame,
                                bud: pd.DataFrame, out_dir: Path) -> Path:
    """One-glance comparison of the three ingested sources."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    # (a) row counts (log scale — they differ by orders of magnitude)
    names = ["Bagrut\n(exam records)", "CBS SES\n(localities)", "Budget\n(institutions)"]
    counts = [len(bag), len(ses), len(bud)]
    ax = axes[0]
    bars = ax.bar(names, counts, color=[NAVY, TEAL, GOLD])
    ax.set_yscale("log")
    ax.set_title("Dataset scale (log)")
    ax.set_ylabel("rows")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center",
                va="bottom", fontsize=12, fontweight="bold")

    # (b) join-key cardinality — shows the two DIFFERENT join keys
    ax = axes[1]
    keys = ["Bagrut\nschools\n(semel)", "Bagrut\ncities\n(name)",
            "CBS\nlocalities\n(name)", "Budget\nschools\n(semel)"]
    vals = [bag["semel"].nunique(), bag["city_norm"].nunique(),
            ses["locality_norm"].nunique(), bud["semel"].nunique()]
    bars = ax.bar(keys, vals, color=[NAVY, NAVY, TEAL, GOLD])
    ax.set_title("Join-key cardinality")
    ax.set_ylabel("distinct values")
    for b, c in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")

    # (c) feature-space contribution from each source
    ax = axes[2]
    contrib = {"Bagrut\n(targets)": 4, "CBS\n(municipal)": 4, "Budget\n(school-level)": 14}
    bars = ax.bar(list(contrib), list(contrib.values()), color=[NAVY, TEAL, GOLD])
    ax.set_title("Columns carried forward")
    ax.set_ylabel("columns")
    for b, c in zip(bars, contrib.values()):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center",
                va="bottom", fontsize=12, fontweight="bold")

    fig.suptitle("Step 1 — Three ingested datasets at a glance", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir, "three_dataset_overview.png")


def plot_budget_column_coverage(bud: pd.DataFrame, cfg: dict[str, Any],
                                out_dir: Path) -> Path:
    """Usable-data audit of every budget column we carried forward."""
    cols = [c for c in cfg["budget"]["numeric_columns"] if c in bud.columns
            and c not in ("semel", "authority_code")]
    rows = []
    for c in cols:
        v = pd.to_numeric(bud[c], errors="coerce")
        rows.append({"column": c, "usable_pct": 100.0 * (v.fillna(0) != 0).mean()})
    d = pd.DataFrame(rows).sort_values("usable_pct", ascending=True)

    colors = [CORAL if p < 5 else (GOLD if p < 60 else TEAL) for p in d["usable_pct"]]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(d["column"], d["usable_pct"], color=colors)
    for y, p in enumerate(d["usable_pct"]):
        ax.text(p + 1, y, f"{p:.1f}%", va="center", fontsize=11)
    ax.set_xlim(0, 108)
    ax.axvline(60, color="grey", ls="--", lw=1)
    ax.set_xlabel("% of institutions with a non-zero value")
    ax.set_title("Budget columns — usable-data audit\n"
                 "(teal ≥60% · gold partial · red unusable)", fontsize=15)
    return _save(fig, out_dir, "budget_column_coverage.png")


def plot_ses_cluster_frequency(ses: pd.DataFrame, out_dir: Path) -> Path:
    """Distribution of the CBS socioeconomic cluster (1-10)."""
    d = ses["cluster"].dropna().astype(int)
    fig, ax = plt.subplots(figsize=(11, 6))
    counts = d.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=TEAL)
    for x, c in zip(range(len(counts)), counts.values):
        ax.text(x, c, str(c), ha="center", va="bottom", fontsize=11)
    ax.set_title("CBS socioeconomic cluster distribution (localities)")
    ax.set_xlabel("cluster (1 = lowest … 10 = highest)")
    ax.set_ylabel("localities")
    return _save(fig, out_dir, "ses_cluster_frequency.png")


def plot_target_missingness(bag: pd.DataFrame, cfg: dict[str, Any],
                            out_dir: Path) -> Path:
    """The 21% grade suppression — and the proof it is NOT missing at random."""
    grade = cfg["bagrut"]["grade_column"]
    takers = cfg["bagrut"]["takers_column"]
    miss = bag[grade].isna()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    vals = [int((~miss).sum()), int(miss.sum())]
    ax.bar(["present", "missing"], vals, color=[TEAL, CORAL])
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:,}\n({100*v/len(bag):.1f}%)", ha="center",
                va="bottom", fontsize=12, fontweight="bold")
    ax.set_title(f"`{grade}` completeness")
    ax.set_ylabel("exam records")

    ax = axes[1]
    data = [bag.loc[~miss, takers].dropna(), bag.loc[miss, takers].dropna()]
    bp = ax.boxplot(data, labels=["grade present", "grade missing"],
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [TEAL, CORAL]):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    med = [float(np.median(d)) for d in data]
    ax.set_title(f"cohort size by missingness\n(median {med[0]:.0f} vs {med[1]:.0f} takers)")
    ax.set_ylabel("takers in the exam cell")

    fig.suptitle("Bagrut target missingness = privacy suppression of small cohorts "
                 "(NOT missing at random)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir, "bagrut_target_missingness.png")


def plot_budget_school_profile(bud: pd.DataFrame, cfg: dict[str, Any],
                               out_dir: Path) -> Path:
    """The NEW school-level categoricals unlocked by the budget dataset.

    Hebrew category labels are mapped to English via ``cfg['display_labels']``:
    matplotlib renders RTL Hebrew reversed ("יהודי" -> "ידוהי"), which would make
    every tick label unreadable at figure scale.
    """
    labels = cfg.get("display_labels", {})
    panels = [("sector", "Sector"), ("supervision", "Supervision"),
              ("district", "District"), ("nurture_quintile", "Nurture quintile")]
    panels = [(c, t) for c, t in panels if c in bud.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.8))
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, panels):
        vc = bud[col].dropna()
        if col == "nurture_quintile":
            vc = vc.astype(int)
        else:
            vc = vc.map(lambda v: labels.get(col, {}).get(v, v))
        vc = vc.value_counts().sort_values(ascending=True)
        ax.barh([str(i) for i in vc.index], vc.values, color=GOLD)
        for y, v in enumerate(vc.values):
            ax.text(v, y, f" {v:,}", va="center", fontsize=10)
        ax.set_title(f"{title}\n({bud[col].notna().mean()*100:.0f}% coverage)", fontsize=13)
        ax.tick_params(axis="y", labelsize=10)
    fig.suptitle("NEW school-level attributes unlocked by Dataset 3 (budget)",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out_dir, "budget_school_profile.png")
