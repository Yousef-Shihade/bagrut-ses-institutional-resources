"""
explain.py — SHAP explainability + leaderboard/ablation visualisation (v2).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shehade & Shada Esawi

shap_beeswarm / shap_importance / plot_leaderboard are unchanged from v1.
plot_before_after is NEW — visualises the Step-5 ablation study (SES-only vs the
Boruta-selected full feature set) that replaces v1's bolted-on Step 6.
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
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk")
NAVY, TEAL, CORAL, GOLD = "#1b2a4a", "#2a9d8f", "#d1495b", "#e8b23a"


def shap_beeswarm(model, X: pd.DataFrame, target: str, cfg: dict[str, Any],
                  out_dir: Path) -> Path:
    """TreeExplainer beeswarm for one tuned champion model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(cfg["modeling"]["shap_sample"], len(X))
    Xs = X.sample(n=n, random_state=cfg["seed"]) if len(X) > n else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, Xs, show=False, plot_type="dot", max_display=15)
    plt.title(f"SHAP — {target}", fontsize=15)
    plt.tight_layout()
    path = out_dir / f"shap_beeswarm_{target}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def shap_importance(model, X: pd.DataFrame, target: str, cfg: dict[str, Any]) -> pd.Series:
    """Mean |SHAP| per feature (used for the README importance table)."""
    n = min(cfg["modeling"]["shap_sample"], len(X))
    Xs = X.sample(n=n, random_state=cfg["seed"]) if len(X) > n else X
    sv = shap.TreeExplainer(model).shap_values(Xs)
    return pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)


def plot_leaderboard(leaderboards: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Grouped bar of CV R^2 for every model across the four targets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, lb in leaderboards.items():
        for _, r in lb.iterrows():
            rows.append({"target": target, "model": r["model"], "R2": r["R2"]})
    long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="model", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Step 5 — Cross-Validated R² Leaderboard (GroupKFold by school)\n"
                 "Full Boruta-selected SES+Budget feature set")
    ax.set_xlabel(""); ax.set_ylabel("CV R²  (higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in long["target"].unique()], fontsize=11)
    ax.legend(title="Model", fontsize=10, loc="upper right")
    plt.tight_layout()
    path = out_dir / "models_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_before_after(ablation: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar: tuned-HGB R^2, SES-only baseline vs Boruta-selected full set
    (same rows, same protocol) — the Step-5 ablation study."""
    out_dir.mkdir(parents=True, exist_ok=True)
    long = ablation.melt(id_vars="target", value_vars=["R2_before", "R2_after"],
                         var_name="phase", value_name="R2")
    long["phase"] = long["phase"].map({"R2_before": "SES only (v1 baseline)",
                                       "R2_after": "SES + Budget (Boruta-selected)"})
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="phase", palette=[NAVY, TEAL], ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Ablation — does the budget dataset add information beyond SES?\n"
                 "(identical rows, tuned HistGradientBoosting, GroupKFold by school)")
    ax.set_xlabel(""); ax.set_ylabel("CV R²  (higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in ablation["target"]], fontsize=11)
    ax.legend(title="", fontsize=11, loc="upper right")
    plt.tight_layout()
    path = out_dir / "ablation_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vif_pruning(vif_result: dict[str, Any], out_dir: Path) -> Path:
    """Initial VIF (with the offending features highlighted) — collinearity story."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vif = vif_result["initial_vif"].sort_values("VIF", ascending=True)
    dropped = set(vif_result["dropped"])
    colors = [CORAL if f in dropped else TEAL for f in vif["feature"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(vif["feature"], vif["VIF"], color=colors)
    ax.axvline(vif_result["threshold"], color=GOLD, ls="--", lw=1.5,
              label=f"threshold = {vif_result['threshold']}")
    for y, v in enumerate(vif["VIF"]):
        ax.text(v, y, f" {v:.1f}", va="center", fontsize=10)
    ax.set_xlabel("Variance Inflation Factor"); ax.legend(fontsize=10)
    ax.set_title("Collinearity — initial VIF (red = dropped by iterative pruning)",
                fontsize=14)
    plt.tight_layout()
    path = out_dir / "vif_pruning.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
