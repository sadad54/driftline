"""PSI and KS drift statistics per feature, computed weekly across the six-month replay, vs. a
week-1 reference distribution.

Why manual PSI/KS (scipy/numpy) instead of Evidently's DataDriftPreset: investigated Evidently
0.7.21's current API (a full rewrite from the `evidently.report.Report` /
`evidently.metric_preset.DataDriftPreset` surface this project's source doc had in mind when
written -- that import path no longer exists). The new preset auto-selects ONE statistical test
per column (KS or Wasserstein for numeric, chi-square or PSI for categorical) rather than
computing both PSI AND KS explicitly per feature, which is what's actually needed here. Getting
both per feature through Evidently would mean instantiating a `ValueDrift(column, method=...)`
metric twice per column for ~370 numeric columns -- more surface area for an unfamiliar new
major-version API to get subtly wrong under time pressure than a direct, well-understood
implementation of two textbook formulas. A real trade-off, made and stated, not hidden.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from driftline.data import load_ieee_cis
from driftline.graph import numeric_feature_columns

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "drift_monitoring.json"
PSI_RETRAIN_THRESHOLD = 0.2
N_FEATURES_TRIGGER = 20  # retrain trigger: this many features crossing PSI > 0.2 in one week


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Standard quantile-binned PSI: bin edges fit on the REFERENCE distribution only, then
    both distributions are binned into those same edges. Population Stability Index =
    sum((cur% - ref%) * ln(cur% / ref%)) across bins, with a small epsilon to avoid log(0)."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return np.nan

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))
    if len(bin_edges) < 3:  # degenerate (near-constant) feature -- PSI undefined/meaningless
        return np.nan

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = ref_counts / ref_counts.sum() + 1e-6
    cur_pct = cur_counts / cur_counts.sum() + 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    print("Loading IEEE-CIS...")
    df = load_ieee_cis()
    num_cols = numeric_feature_columns(df)
    print(f"  {len(num_cols)} numeric features monitored")

    dt_min = df["TransactionDT"].min()
    df = df.copy()
    df["week"] = (df["TransactionDT"] - dt_min) // (7 * 86400)
    n_weeks = int(df["week"].max()) + 1
    print(f"  {n_weeks} weeks across the replay span")

    reference = df[df["week"] == 0]
    print(f"  reference (week 0): {len(reference):,} rows")

    weekly_results = {}
    first_trigger_week = None
    for week in range(1, n_weeks):
        current = df[df["week"] == week]
        if len(current) < 50:
            continue

        psi_values, ks_values = {}, {}
        for col in num_cols:
            ref_vals = reference[col].to_numpy(dtype=np.float64)
            cur_vals = current[col].to_numpy(dtype=np.float64)
            psi_values[col] = compute_psi(ref_vals, cur_vals)
            ref_clean = ref_vals[~np.isnan(ref_vals)]
            cur_clean = cur_vals[~np.isnan(cur_vals)]
            if len(ref_clean) >= 10 and len(cur_clean) >= 10:
                ks_values[col] = float(ks_2samp(ref_clean, cur_clean).statistic)
            else:
                ks_values[col] = np.nan

        psi_series = pd.Series(psi_values)
        n_breach = int((psi_series > PSI_RETRAIN_THRESHOLD).sum())
        weekly_results[f"week_{week}"] = {
            "n_rows": len(current),
            "n_features_psi_breach": n_breach,
            "mean_psi": float(psi_series.mean(skipna=True)),
            "max_psi": float(psi_series.max(skipna=True)),
            "max_psi_feature": psi_series.idxmax(),
            "mean_ks": float(pd.Series(ks_values).mean(skipna=True)),
        }
        print(f"  week {week}: {n_breach} features PSI>{PSI_RETRAIN_THRESHOLD} "
              f"(max={psi_series.max(skipna=True):.3f} on {psi_series.idxmax()}), "
              f"mean_ks={pd.Series(ks_values).mean(skipna=True):.3f}")

        if first_trigger_week is None and n_breach >= N_FEATURES_TRIGGER:
            first_trigger_week = week
            print(f"    *** retrain trigger crossed: {n_breach} >= {N_FEATURES_TRIGGER} features ***")

    summary = {
        "n_weeks": n_weeks,
        "retrain_threshold_psi": PSI_RETRAIN_THRESHOLD,
        "retrain_trigger_n_features": N_FEATURES_TRIGGER,
        "first_trigger_week": first_trigger_week,
        "weekly": weekly_results,
    }
    print(f"\nFirst week crossing the retrain trigger ({N_FEATURES_TRIGGER}+ features, "
          f"PSI>{PSI_RETRAIN_THRESHOLD}): week {first_trigger_week}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
