"""
imputation_experiment.py — Task A: MICE robustness (v2: multi-iteration).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

v1 ran the "mask 8% and reconstruct" test ONCE. The lecturer flagged this as
insufficient evidence: a single lucky/unlucky random mask does not prove the
method is *stable*. v2 repeats the experiment N_ITERATIONS times, each with an
independent random mask (different seed, same 8% fraction, same predictor set),
and reports the distribution (mean +/- std, min/max) of R^2/RMSE/MAE across runs
for both MICE and the median baseline — proving the result generalises rather
than being a one-off draw.
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

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

sns.set_theme(style="whitegrid", context="talk")
NAVY, TEAL, CORAL, GOLD = "#1b2a4a", "#2a9d8f", "#d1495b", "#e8b23a"


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def _run_once(X: pd.DataFrame, target: str, mask_fraction: float,
              mice_max_iter: int, seed: int) -> dict[str, Any]:
    """One masked-reconstruction trial with a given random seed."""
    truth = X[target].to_numpy().copy()
    rng = np.random.default_rng(seed)
    n = len(X)
    n_mask = int(round(n * mask_fraction))
    mask_idx = rng.choice(n, size=n_mask, replace=False)
    mask_bool = np.zeros(n, dtype=bool)
    mask_bool[mask_idx] = True

    X_masked = X.copy()
    X_masked.loc[mask_bool, target] = np.nan
    true_masked = truth[mask_bool]

    mice = IterativeImputer(estimator=BayesianRidge(), max_iter=mice_max_iter,
                            random_state=seed, sample_posterior=False)
    mice_full = mice.fit_transform(X_masked)
    mice_pred = mice_full[:, X.columns.get_loc(target)][mask_bool]

    med = SimpleImputer(strategy="median")
    median_col = med.fit_transform(X_masked[[target]]).ravel()
    median_pred = median_col[mask_bool]

    return {"seed": seed, "n_masked": n_mask,
            "mice": _metrics(mice_pred, true_masked),
            "median": _metrics(median_pred, true_masked)}


def run_multi_iteration(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the masking experiment N_ITERATIONS times with independent seeds."""
    ie = cfg["imputation_experiment"]
    target = ie["target_feature"]
    predictors = ie["predictors"]
    n_iter = ie["n_iterations"]
    base_seed = cfg["seed"]

    data = df[df[target].notna()].copy()
    X = data[predictors].astype(float).reset_index(drop=True)

    rows = []
    for i in range(n_iter):
        seed = base_seed + i
        res = _run_once(X, target, ie["mask_fraction"], ie["mice_max_iter"], seed)
        rows.append({"run": i, "seed": seed,
                    "MICE_R2": res["mice"]["R2"], "MICE_RMSE": res["mice"]["RMSE"],
                    "MICE_MAE": res["mice"]["MAE"],
                    "Median_R2": res["median"]["R2"], "Median_RMSE": res["median"]["RMSE"],
                    "Median_MAE": res["median"]["MAE"]})

    runs = pd.DataFrame(rows)
    summary = {
        "n_iterations": n_iter, "n_total": len(X),
        "n_masked": int(round(len(X) * ie["mask_fraction"])),
        "target": target, "predictors": predictors,
        "MICE_R2_mean": float(runs["MICE_R2"].mean()), "MICE_R2_std": float(runs["MICE_R2"].std()),
        "MICE_R2_min": float(runs["MICE_R2"].min()), "MICE_R2_max": float(runs["MICE_R2"].max()),
        "MICE_RMSE_mean": float(runs["MICE_RMSE"].mean()), "MICE_RMSE_std": float(runs["MICE_RMSE"].std()),
        "Median_R2_mean": float(runs["Median_R2"].mean()), "Median_R2_std": float(runs["Median_R2"].std()),
        "Median_RMSE_mean": float(runs["Median_RMSE"].mean()), "Median_RMSE_std": float(runs["Median_RMSE"].std()),
    }
    return {"runs": runs, "summary": summary}


def plot_robustness(result: dict[str, Any], out_dir: Path) -> Path:
    """Boxplot of R^2 across all iterations, MICE vs median — the stability proof."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, summ = result["runs"], result["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: R^2 distribution across runs (the stability evidence).
    box_data = [runs["MICE_R2"], runs["Median_R2"]]
    bp = ax1.boxplot(box_data, labels=["MICE", "Median"], patch_artist=True,
                     widths=0.5)
    for patch, c in zip(bp["boxes"], [TEAL, CORAL]):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, data in enumerate(box_data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, len(data))
        ax1.scatter(np.full(len(data), i) + jitter, data, s=18, color="black",
                   alpha=0.5, zorder=3)
    ax1.set_title(f"R² across {summ['n_iterations']} independent runs\n"
                 f"MICE {summ['MICE_R2_mean']:.3f} ± {summ['MICE_R2_std']:.3f}  vs  "
                 f"Median {summ['Median_R2_mean']:.3f} ± {summ['Median_R2_std']:.3f}",
                 fontsize=13)
    ax1.set_ylabel("R² (reconstruction of masked cells)")

    # Right: per-run R^2 trend line — visually shows no drift/instability.
    ax2.plot(runs["run"], runs["MICE_R2"], "o-", color=TEAL, label="MICE", lw=1.5)
    ax2.plot(runs["run"], runs["Median_R2"], "o-", color=CORAL, label="Median", lw=1.5)
    ax2.axhline(summ["MICE_R2_mean"], color=TEAL, ls="--", lw=1, alpha=0.6)
    ax2.set_title("R² per run (different random 8% mask each time)")
    ax2.set_xlabel("run index"); ax2.set_ylabel("R²")
    ax2.legend(fontsize=11)

    fig.suptitle(f"Task A — MICE Robustness: {summ['n_iterations']} independent masking trials "
                f"({int(summ['n_masked'])} cells masked each run, 8%)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "mice_robustness_multi_iteration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
