"""
run_step4.py — Step 4 orchestrator.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Runs the three Step-4 tasks end to end and writes the modeling-ready cache:
    Task A — MICE vs median imputation experiment
    Task B — Isolation Forest + LOF outlier detection
    Task C — Q1 subject resilience gap, Q2 low-SES overachievers
    -> data/cleaned_modeling_ready.csv (+ 4 plots) + console summary

Usage:
    python code/run_step4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import exploratory  # noqa: E402
import imputation_experiment as imp  # noqa: E402
import outliers as ol  # noqa: E402
from io_load import load_config, load_school_level, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 74


def main() -> None:
    cfg = load_config()
    graphs = resolve(cfg["paths"]["out_graphs"])
    plots: list[Path] = []

    print(_hr())
    print("STEP 4 — PREPROCESSING, OUTLIER DETECTION & IMPUTATION EXPERIMENT")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    df = load_school_level(cfg)

    # ===================== Task A — MICE imputation ======================= #
    impr = imp.run_experiment(df, cfg)
    plots.append(imp.plot_comparison(impr, graphs))
    print("\n[TASK A] MICE vs MEDIAN IMPUTATION  (feature = "
          f"'{impr['target']}', {int(impr['mask_fraction']*100)}% of "
          f"{impr['n_total']} rows masked = {impr['n_masked']} cells)")
    print("    method   |    RMSE      MAE       R^2")
    print(f"    MICE     |  {impr['mice']['RMSE']:7.4f}  {impr['mice']['MAE']:7.4f}  "
          f"{impr['mice']['R2']:7.4f}")
    print(f"    Median   |  {impr['median']['RMSE']:7.4f}  {impr['median']['MAE']:7.4f}  "
          f"{impr['median']['R2']:7.4f}")
    rmse_gain = (1 - impr["mice"]["RMSE"] / impr["median"]["RMSE"]) * 100
    print(f"    -> MICE reduces RMSE by {rmse_gain:.1f}% vs the median baseline.")

    # ===================== Task B — Outliers ============================== #
    df, osum = ol.detect(df, cfg)
    plots.append(ol.plot_mapping(df, graphs))
    print(f"\n[TASK B] OUTLIER DETECTION  (n_evaluated = {osum['n_evaluated']}, "
          f"contamination = {cfg['outliers']['contamination']})")
    print(f"    Isolation Forest flagged : {osum['iso_flagged']}")
    print(f"    LOF flagged              : {osum['lof_flagged']}")
    print(f"    consensus (both)         : {osum['overlap_both']}  "
          f"(Jaccard = {osum['jaccard']:.2f})")
    print(f"    flagged by either        : {osum['any_flagged']}  "
          f"[iso-only {osum['iso_only']}, lof-only {osum['lof_only']}]")

    # ===================== Task C — Exploratory =========================== #
    q1 = exploratory.question1_resilience(df, cfg)
    plots.append(exploratory.plot_resilience_gap(q1, graphs))
    print(f"\n[TASK C — Q1] SUBJECT RESILIENCE  (cluster {q1['low_cluster']} vs "
          f"{q1['high_cluster']} gap; smaller = more resilient)")
    for s, v in q1["subjects"].items():
        print(f"    {s:8s} grade {v['grade_lo']:5.1f}->{v['grade_hi']:5.1f} "
              f"gap={v['grade_gap']:+5.2f}pts ({v['grade_gap_pct']:+4.1f}%, d={v['grade_cohens_d']:.2f}) | "
              f"5u-part {v['part_lo']:.3f}->{v['part_hi']:.3f} gap={v['part_gap']:+.3f}")
    print(f"    => MORE RESILIENT (grade gap): {q1['more_resilient_grade']};  "
          f"(participation gap): {q1['more_resilient_participation']}")

    q2 = exploratory.question2_overachievers(df, cfg)
    plots.append(exploratory.plot_overachievers(q2, graphs))
    print(f"\n[TASK C — Q2] LOW-SES OVERACHIEVERS  (clusters {q2['low_clusters']} "
          f"matching elite {q2['elite_clusters']} {q2['benchmark_type']} grade "
          f"{q2['benchmark']:.2f})")
    print(f"    overachievers: {q2['n_over']}/{q2['n_low']} low-SES schools "
          f"({q2['over_pct']:.1f}%)")
    print(f"    advanced-track selection  Math 5u : {q2['over_math_part']:.3f} "
          f"(over) vs {q2['normal_math_part']:.3f} (normal)  p={q2['p_math_part']:.1e}")
    print(f"                              Eng  5u : {q2['over_eng_part']:.3f} "
          f"(over) vs {q2['normal_eng_part']:.3f} (normal)  p={q2['p_eng_part']:.1e}")
    print("    top overachievers (school | cluster | combined grade):")
    for _, r in q2["top"].head(5).iterrows():
        print(f"      - {r['school'][:28]:28s} cl{int(r['cluster'])}  {r['combined']:.1f}")

    # ===================== Cleaned modeling-ready cache =================== #
    cleaned = df[df["cluster"].notna()].copy()
    resolve(cfg["paths"]["out_data"]).mkdir(parents=True, exist_ok=True)
    out_path = resolve(cfg["paths"]["cleaned_out"])
    cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n[OUTPUT] CLEANED MODELING-READY DATASET")
    print(f"    {out_path.name}: {cleaned.shape[0]:,} rows x {cleaned.shape[1]} cols "
          "(cluster-present school-years)")
    print(f"    outliers FLAGGED not dropped -> 'outlier_consensus' True: "
          f"{int(cleaned['outlier_consensus'].sum())} "
          f"(Step 5 can exclude via this flag)")
    print(f"    targets retain NaN (suppression) — never imputed.")
    print("    graphs:")
    for p in plots:
        print(f"      - {Path(p).name:34s} ({Path(p).stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 4 COMPLETE ✔   (awaiting signal for Step 5)")
    print(_hr())


if __name__ == "__main__":
    main()
