"""
exploratory.py — Task C: baseline answers to the two core research questions.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Q1 (Subject resilience): which subject's cluster-2 -> cluster-9 gap is smallest?
Q2 (Low-SES overachievers): which low-cluster (1-4) schools match elite (8-10)
    grades, and how does their advanced-track selection differ?
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="talk")


# --------------------------------------------------------------------------- #
# Q1 — Subject resilience gap (cluster 2 vs cluster 9)
# --------------------------------------------------------------------------- #
def question1_resilience(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    ex = cfg["exploratory"]
    lo, hi = ex["resilience_low_cluster"], ex["resilience_high_cluster"]
    res: dict[str, Any] = {"low_cluster": lo, "high_cluster": hi, "subjects": {}}

    pairs = {
        "Math": ("math_avg_grade", "math_5unit_participation"),
        "English": ("english_avg_grade", "english_5unit_participation"),
    }
    for subj, (gcol, pcol) in pairs.items():
        g_lo = df.loc[df["cluster"] == lo, gcol].dropna()
        g_hi = df.loc[df["cluster"] == hi, gcol].dropna()
        p_lo = df.loc[df["cluster"] == lo, pcol].dropna()
        p_hi = df.loc[df["cluster"] == hi, pcol].dropna()
        # Cohen's d for the grade gap (standardised, comparable across subjects).
        pooled_sd = np.sqrt((g_lo.var(ddof=1) + g_hi.var(ddof=1)) / 2)
        d = (g_hi.mean() - g_lo.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        res["subjects"][subj] = {
            "grade_lo": g_lo.mean(), "grade_hi": g_hi.mean(),
            "grade_gap": g_hi.mean() - g_lo.mean(),
            "grade_gap_pct": (g_hi.mean() - g_lo.mean()) / g_lo.mean() * 100,
            "grade_cohens_d": d,
            "part_lo": p_lo.mean(), "part_hi": p_hi.mean(),
            "part_gap": p_hi.mean() - p_lo.mean(),
        }
    res["more_resilient_grade"] = min(res["subjects"],
                                      key=lambda s: res["subjects"][s]["grade_gap"])
    res["more_resilient_participation"] = min(
        res["subjects"], key=lambda s: res["subjects"][s]["part_gap"])
    return res


def plot_resilience_gap(res: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subs = list(res["subjects"].keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    colors = {"Math": "#3b6ea5", "English": "#c44e52"}

    # Grade gap (points).
    gaps = [res["subjects"][s]["grade_gap"] for s in subs]
    ax1.bar(subs, gaps, color=[colors[s] for s in subs])
    for i, s in enumerate(subs):
        ax1.text(i, gaps[i], f"{gaps[i]:.2f} pts\n(d={res['subjects'][s]['grade_cohens_d']:.2f})",
                 ha="center", va="bottom", fontsize=12)
    ax1.set_title(f"Grade gap (cluster {res['high_cluster']} − {res['low_cluster']})")
    ax1.set_ylabel("Avg grade gap (points)")
    ax1.margins(y=0.18)

    # Participation gap (rate).
    pgaps = [res["subjects"][s]["part_gap"] for s in subs]
    ax2.bar(subs, pgaps, color=[colors[s] for s in subs])
    for i, s in enumerate(subs):
        ax2.text(i, pgaps[i], f"{pgaps[i]:.3f}", ha="center", va="bottom", fontsize=12)
    ax2.set_title(f"5-unit participation gap (cluster {res['high_cluster']} − {res['low_cluster']})")
    ax2.set_ylabel("Participation-rate gap")
    ax2.margins(y=0.18)

    fig.suptitle("Q1 — Subject Resilience to Socioeconomic Disparity "
                 f"(smaller gap = more resilient → {res['more_resilient_grade']})",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "subject_resilience_gap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Q2 — Low-SES overachievers
# --------------------------------------------------------------------------- #
def question2_overachievers(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    ex = cfg["exploratory"]
    low_cl, elite_cl = ex["overachiever_low_clusters"], ex["overachiever_elite_clusters"]

    # Aggregate to school level (mean across years) so "consistently" is captured.
    sch = df.groupby("semel").agg(
        cluster=("cluster", "first"),
        combined=("combined_avg_grade", "mean"),
        math_part=("math_5unit_participation", "mean"),
        eng_part=("english_5unit_participation", "mean"),
        school=("school", "first"), city=("city", "first"),
    ).dropna(subset=["cluster", "combined"])

    bench_func = ex["overachiever_benchmark"]
    elite = sch[sch["cluster"].isin(elite_cl)]["combined"]
    benchmark = float(getattr(elite, bench_func)())

    low = sch[sch["cluster"].isin(low_cl)].copy()
    low["overachiever"] = low["combined"] >= benchmark
    over = low[low["overachiever"]]
    normal = low[~low["overachiever"]]

    # Significance of the participation difference (overachievers vs normal low-SES).
    def _ttest(a, b):
        a, b = a.dropna(), b.dropna()
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            return float(p)
        return np.nan

    return {
        "benchmark": benchmark, "benchmark_type": bench_func,
        "low_clusters": low_cl, "elite_clusters": elite_cl,
        "n_low": len(low), "n_over": len(over),
        "over_pct": len(over) / len(low) * 100 if len(low) else 0,
        "over_math_part": over["math_part"].mean(), "over_eng_part": over["eng_part"].mean(),
        "normal_math_part": normal["math_part"].mean(),
        "normal_eng_part": normal["eng_part"].mean(),
        "p_math_part": _ttest(over["math_part"], normal["math_part"]),
        "p_eng_part": _ttest(over["eng_part"], normal["eng_part"]),
        "_low_df": low, "_over_df": over,
        "top": over.sort_values("combined", ascending=False)
                   .head(8)[["school", "city", "cluster", "combined",
                             "math_part", "eng_part"]],
    }


def plot_overachievers(res: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    low = res["_low_df"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: low-SES schools, grade vs cluster, overachievers highlighted.
    jitter = np.random.default_rng(0).normal(0, 0.07, len(low))
    normal = low[~low["overachiever"]]; over = low[low["overachiever"]]
    ax1.scatter(normal["cluster"] + jitter[~low["overachiever"].values],
                normal["combined"], s=26, color="#9bb8d3", alpha=0.6,
                label=f"Normal low-SES (n={len(normal)})")
    ax1.scatter(over["cluster"] + jitter[low["overachiever"].values],
                over["combined"], s=70, color="#d1495b", edgecolor="black",
                linewidth=0.4, label=f"Overachievers (n={len(over)})", zorder=3)
    ax1.axhline(res["benchmark"], color="black", ls="--", lw=1.5,
                label=f"Elite (8-10) {res['benchmark_type']} = {res['benchmark']:.1f}")
    ax1.set_title("Low-SES schools (clusters 1-4) vs elite benchmark")
    ax1.set_xlabel("CBS socioeconomic cluster")
    ax1.set_ylabel("School mean combined grade")
    ax1.legend(fontsize=10, loc="lower left")

    # Right: advanced-track selection profile, overachievers vs normal.
    cats = ["Math 5-unit", "English 5-unit"]
    over_vals = [res["over_math_part"], res["over_eng_part"]]
    norm_vals = [res["normal_math_part"], res["normal_eng_part"]]
    x = np.arange(len(cats)); w = 0.38
    ax2.bar(x - w / 2, over_vals, w, color="#d1495b", label="Overachievers")
    ax2.bar(x + w / 2, norm_vals, w, color="#9bb8d3", label="Normal low-SES")
    for i in range(len(cats)):
        ax2.text(i - w / 2, over_vals[i], f"{over_vals[i]:.2f}", ha="center", va="bottom", fontsize=11)
        ax2.text(i + w / 2, norm_vals[i], f"{norm_vals[i]:.2f}", ha="center", va="bottom", fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(cats)
    ax2.set_title("Advanced-track participation profile")
    ax2.set_ylabel("Mean 5-unit participation rate")
    ax2.legend(fontsize=11); ax2.margins(y=0.18)

    fig.suptitle("Q2 — Low-SES Overachievers: who matches elite grades & how they select",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "low_ses_overachievers_profile.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
