"""
outliers.py — Task B: Isolation Forest + Local Outlier Factor.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Flags anomalous school-year records (uncharacteristic grade/SES/volume profiles)
with two complementary detectors and contrasts them:
  * Isolation Forest — global, tree-isolation based.
  * Local Outlier Factor — local-density based.
Both run on the same standardised feature space with the same contamination, so
their disagreement is informative (global vs local notion of "anomaly").
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

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="talk")


def detect(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (df + flag columns, summary). Flags only on complete-case rows."""
    oc = cfg["outliers"]
    feats = oc["features"]
    seed = cfg["seed"]

    out = df.copy()
    out["iso_outlier"] = False
    out["lof_outlier"] = False
    out["lof_score"] = np.nan
    out["iso_score"] = np.nan

    # Complete-case rows in the detection feature space.
    mask = df[feats].notna().all(axis=1)
    X = df.loc[mask, feats].astype(float)
    Xs = StandardScaler().fit_transform(X)

    # Isolation Forest (predict: -1 = outlier).
    iso = IsolationForest(contamination=oc["contamination"], random_state=seed,
                          n_estimators=300)
    iso_pred = iso.fit_predict(Xs)
    out.loc[mask, "iso_outlier"] = iso_pred == -1
    out.loc[mask, "iso_score"] = iso.score_samples(Xs)   # higher = more normal

    # Local Outlier Factor (fit_predict: -1 = outlier).
    lof = LocalOutlierFactor(n_neighbors=oc["lof_neighbors"],
                             contamination=oc["contamination"])
    lof_pred = lof.fit_predict(Xs)
    out.loc[mask, "lof_outlier"] = lof_pred == -1
    out.loc[mask, "lof_score"] = lof.negative_outlier_factor_  # lower = more outlying

    out["outlier_consensus"] = out["iso_outlier"] & out["lof_outlier"]
    out["outlier_any"] = out["iso_outlier"] | out["lof_outlier"]

    n_eval = int(mask.sum())
    iso_n = int(out["iso_outlier"].sum())
    lof_n = int(out["lof_outlier"].sum())
    both = int(out["outlier_consensus"].sum())
    summary = {
        "n_evaluated": n_eval, "features": feats,
        "iso_flagged": iso_n, "lof_flagged": lof_n,
        "overlap_both": both, "any_flagged": int(out["outlier_any"].sum()),
        "jaccard": both / (iso_n + lof_n - both) if (iso_n + lof_n - both) > 0 else 0.0,
        "iso_only": iso_n - both, "lof_only": lof_n - both,
    }
    return out, summary


def plot_mapping(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter (SES index vs combined grade) colour-coded by detector agreement."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d = df[df[["index_value", "combined_avg_grade"]].notna().all(axis=1)].copy()

    def _cat(r):
        if r["iso_outlier"] and r["lof_outlier"]:
            return "Both (consensus)"
        if r["iso_outlier"]:
            return "Isolation Forest only"
        if r["lof_outlier"]:
            return "LOF only"
        return "Clean"
    d["flag"] = d.apply(_cat, axis=1)

    palette = {"Clean": "#c7c7c7", "Isolation Forest only": "#3b6ea5",
               "LOF only": "#e9a23b", "Both (consensus)": "#d1495b"}
    order = ["Clean", "Isolation Forest only", "LOF only", "Both (consensus)"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for cat in order:
        sub = d[d["flag"] == cat]
        ax.scatter(sub["index_value"], sub["combined_avg_grade"],
                   s=(18 if cat == "Clean" else 55),
                   c=palette[cat], alpha=(0.35 if cat == "Clean" else 0.9),
                   edgecolor=("none" if cat == "Clean" else "black"), linewidth=0.4,
                   label=f"{cat} (n={len(sub)})", zorder=(1 if cat == "Clean" else 3))
    ax.set_title("Task B — Outlier Mapping: Isolation Forest vs LOF\n"
                 "(school-year records in SES-vs-grade space)")
    ax.set_xlabel("CBS socioeconomic index value (higher = wealthier)")
    ax.set_ylabel("Combined (Math+English) weighted avg grade")
    ax.legend(fontsize=11, loc="lower right")
    path = out_dir / "outlier_detection_mapping.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
