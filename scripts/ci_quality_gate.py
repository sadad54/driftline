"""CI quality gate: train XGBoost on the small committed CI sample (tests/fixtures/), compare
PR-AUC against a checked-in baseline, fail the build if it regresses by more than the tolerance.

This is real numbers from a real (if small) train/eval run every time CI executes it -- not a
hardcoded pass. The baseline itself (ci_baseline.json) is regenerated deliberately when the
model/features/data genuinely change, not silently overwritten by this script.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from driftline.baseline import evaluate, train_xgboost
from driftline.data import feature_columns, time_ordered_split

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ieee_cis_ci_sample.parquet"
BASELINE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ci_baseline.json"
REGRESSION_TOLERANCE = 0.01  # PR-AUC may not drop by more than this (absolute) vs. baseline


def main():
    df = pd.read_parquet(FIXTURE_PATH)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")

    train, test = time_ordered_split(df, test_frac=0.2)
    cols = feature_columns(df)

    print(f"Training on CI sample: {len(train)} train / {len(test)} test rows")
    _, scores = train_xgboost(train[cols], train["isFraud"], test[cols])
    metrics = evaluate(test["isFraud"].to_numpy(), scores)
    print(f"Current PR-AUC: {metrics['pr_auc']:.4f}")

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    baseline_pr_auc = baseline["pr_auc"]
    print(f"Baseline PR-AUC: {baseline_pr_auc:.4f}")

    regression = baseline_pr_auc - metrics["pr_auc"]
    print(f"Regression: {regression:+.4f} (tolerance: {REGRESSION_TOLERANCE})")

    if regression > REGRESSION_TOLERANCE:
        print(f"\nFAIL: PR-AUC regressed by {regression:.4f}, exceeding tolerance {REGRESSION_TOLERANCE}")
        sys.exit(1)
    print("\nPASS: no significant PR-AUC regression")


if __name__ == "__main__":
    main()
