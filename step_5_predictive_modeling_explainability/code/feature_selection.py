"""
feature_selection.py — collinearity (iterative VIF) + Boruta selection.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Two concerns are handled here:
  * Collinearity: with 15 numeric candidates, spotting redundant pairs by hand
    does not scale, so collinearity handling is an ITERATIVE procedure: compute
    VIF for all candidates, drop the single worst offender, recompute, repeat
    until every remaining feature is below the threshold. Iteration matters
    because dropping one feature can resolve another's inflation — a single-pass
    cutoff would discard features that are only collinear via a third.
  * Feature selection: Boruta (an all-relevant random-forest wrapper) runs on
    the VIF-pruned SES+budget candidate set, once per target.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compatibility shim: some Boruta builds reference removed NumPy aliases.
for _a, _t in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
from boruta import BorutaPy  # noqa: E402


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF for every column in X (X must be numeric, complete-case, with intercept)."""
    Xc = X.assign(_const=1.0)
    rows = []
    for i, c in enumerate(Xc.columns):
        if c == "_const":
            continue
        rows.append({"feature": c, "VIF": variance_inflation_factor(Xc.values, i)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def iterative_vif_prune(df: pd.DataFrame, candidates: list[str],
                        threshold: float = 5.0) -> dict[str, Any]:
    """Repeatedly drop the highest-VIF feature until all remaining are below
    ``threshold`` (or 2 features remain, the minimum needed for a meaningful VIF).

    Returns: kept, dropped (in drop order), history (VIF table at each step),
    initial_vif (the first full table, for reporting).
    """
    remaining = list(candidates)
    X = df[remaining].dropna().astype(float)

    history: list[dict[str, Any]] = []
    dropped: list[str] = []
    initial_vif = _vif_table(X[remaining]) if len(remaining) > 1 else pd.DataFrame()

    while len(remaining) > 2:
        vif = _vif_table(X[remaining])
        worst = vif.iloc[0]
        if worst["VIF"] < threshold:
            break
        history.append({"step": len(dropped) + 1, "dropped_feature": worst["feature"],
                        "VIF_at_drop": float(worst["VIF"]), "remaining_after": len(remaining) - 1})
        remaining.remove(worst["feature"])
        dropped.append(worst["feature"])

    final_vif = _vif_table(X[remaining]) if len(remaining) > 1 else pd.DataFrame()
    return {"kept": remaining, "dropped": dropped, "history": pd.DataFrame(history),
            "initial_vif": initial_vif, "final_vif": final_vif, "threshold": threshold}


def run_boruta(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run Boruta on one target; return confirmed / tentative / rejected lists."""
    fs = cfg["feature_selection"]
    seed = cfg["seed"]
    rf = RandomForestRegressor(n_estimators=fs["boruta_estimators"],
                               random_state=seed, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators=fs["boruta_estimators"],
                      max_iter=fs["boruta_max_iter"], perc=fs.get("boruta_perc", 90),
                      random_state=seed, verbose=0)
    boruta.fit(X.values.astype(float), y.values.astype(float))

    cols = np.array(X.columns)
    confirmed = cols[boruta.support_].tolist()
    tentative = cols[boruta.support_weak_].tolist()
    rejected = [c for c in cols if c not in confirmed and c not in tentative]
    ranking = dict(zip(cols.tolist(), boruta.ranking_.tolist()))

    # Features actually used downstream: the top Boruta tier (rank 1 = confirmed).
    # If <2 reach rank 1, keep the 3 best-ranked so at least one SES variable
    # (cluster) is always retained rather than modeling on zero features.
    rank1 = [c for c in cols.tolist() if ranking[c] == 1]
    if len(rank1) >= 2:
        selected = rank1
    else:
        selected = [c for c, _ in sorted(ranking.items(), key=lambda kv: kv[1])[:3]]

    return {"confirmed": confirmed, "tentative": tentative, "rejected": rejected,
            "selected": selected, "ranking": ranking}
