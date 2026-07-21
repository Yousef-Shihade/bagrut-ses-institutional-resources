"""
imputation_experiment.py — Task A: MICE vs median imputation (the lecturer's test).

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Procedure (per the presentation guideline "if none missing, artificially remove
5-10% and compare imputation success"):
  1. Take a fully-populated continuous feature (CBS ``index_value``).
  2. Randomly mask `mask_fraction` of its values (8%).
  3. Reconstruct with **MICE** (sklearn ``IterativeImputer``) using the other
     numeric features as predictors.
  4. Reconstruct a parallel copy with a **median** baseline.
  5. Compare RMSE / MAE / R^2 on the masked cells, and overlay the densities.
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

from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables import below)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

sns.set_theme(style="whitegrid", context="talk")


def run_experiment(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Mask 8% of ``index_value`` and compare MICE vs median reconstruction."""
    ie = cfg["imputation_experiment"]
    seed = cfg["seed"]
    target = ie["target_feature"]
    predictors = ie["predictors"]

    # Work on rows where the experiment target is fully observed.
    data = df[df[target].notna()].copy()
    X = data[predictors].astype(float).reset_index(drop=True)
    truth = X[target].to_numpy().copy()

    # --- Randomly mask mask_fraction of the target values --------------------
    rng = np.random.default_rng(seed)
    n = len(X)
    n_mask = int(round(n * ie["mask_fraction"]))
    mask_idx = rng.choice(n, size=n_mask, replace=False)
    mask_bool = np.zeros(n, dtype=bool)
    mask_bool[mask_idx] = True

    X_masked = X.copy()
    X_masked.loc[mask_bool, target] = np.nan

    # --- MICE (IterativeImputer over the full numeric matrix) ----------------
    mice = IterativeImputer(estimator=BayesianRidge(), max_iter=ie["mice_max_iter"],
                            random_state=seed, sample_posterior=False)
    mice_full = mice.fit_transform(X_masked)
    mice_col = mice_full[:, predictors.index(target)]
    mice_pred = mice_col[mask_bool]

    # --- Median baseline (single-column, ignores all structure) --------------
    med = SimpleImputer(strategy="median")
    median_col = med.fit_transform(X_masked[[target]]).ravel()
    median_pred = median_col[mask_bool]
    median_value = float(med.statistics_[0])

    # --- Metrics on the masked cells -----------------------------------------
    true_masked = truth[mask_bool]

    def _metrics(pred: np.ndarray) -> dict[str, float]:
        err = pred - true_masked
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((true_masked - true_masked.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    results = {
        "n_total": n, "n_masked": n_mask, "mask_fraction": ie["mask_fraction"],
        "target": target, "median_value": median_value,
        "mice": _metrics(mice_pred), "median": _metrics(median_pred),
        # Full reconstructed columns for the density plot.
        "_truth_full": truth,
        "_mice_full": mice_col,
        "_median_full": np.where(mask_bool, median_col, truth),
        "_true_masked": true_masked, "_mice_pred": mice_pred, "_median_pred": median_pred,
    }
    return results


def plot_comparison(results: dict[str, Any], out_dir: Path) -> Path:
    """Density overlay: original vs MICE-imputed vs median-imputed full column."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: full-column density (shows the median spike artifact).
    sns.kdeplot(results["_truth_full"], ax=ax1, color="#2a9d8f", lw=2.5,
                label="Original (no missing)", fill=True, alpha=0.15)
    sns.kdeplot(results["_mice_full"], ax=ax1, color="#3b6ea5", lw=2.5,
                label="MICE-imputed", linestyle="--")
    sns.kdeplot(results["_median_full"], ax=ax1, color="#d1495b", lw=2.5,
                label="Median-imputed", linestyle=":")
    ax1.axvline(results["median_value"], color="#d1495b", lw=1, alpha=0.5)
    ax1.set_title(f"Density of {results['target']} after imputation")
    ax1.set_xlabel(results["target"]); ax1.legend(fontsize=11)

    # Right: predicted-vs-true scatter on the masked cells.
    tm = results["_true_masked"]
    ax2.scatter(tm, results["_mice_pred"], s=22, color="#3b6ea5", alpha=0.7,
                label=f"MICE (RMSE={results['mice']['RMSE']:.3f})")
    ax2.scatter(tm, results["_median_pred"], s=22, color="#d1495b", alpha=0.5,
                marker="x", label=f"Median (RMSE={results['median']['RMSE']:.3f})")
    lim = [tm.min() - 0.2, tm.max() + 0.2]
    ax2.plot(lim, lim, color="black", lw=1, ls="--", label="perfect")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_title("Masked cells — predicted vs true")
    ax2.set_xlabel(f"true {results['target']}"); ax2.set_ylabel("imputed value")
    ax2.legend(fontsize=11)

    fig.suptitle("Task A — Imputation Success: MICE vs Median Baseline "
                 f"({int(results['mask_fraction']*100)}% masked)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "imputation_success_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
