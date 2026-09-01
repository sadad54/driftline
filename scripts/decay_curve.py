"""The headline drift artifact: train XGBoost on month 1 only, evaluate on months 1-6 WITHOUT
retraining, and chart the real PR-AUC decay this data actually exhibits -- not injected synthetic
drift. "Month" here means 1/6th of the observed ~182-day TransactionDT span (see
data/README.md); there's no real calendar in this dataset.
"""
import json
from pathlib import Path

import numpy as np

from driftline.baseline import evaluate, train_xgboost
from driftline.data import add_month_bucket, feature_columns, load_ieee_cis

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "decay_curve.json"


def main():
    print("Loading IEEE-CIS and bucketing into 6 months...")
    df = load_ieee_cis()
    df = add_month_bucket(df, n_buckets=6)
    cols = feature_columns(df)

    print(df.groupby("month").size().to_string())

    month1 = df[df["month"] == 1]
    print(f"\nTraining XGBoost on month 1 only ({len(month1):,} rows, "
          f"fraud rate {month1['isFraud'].mean():.4%})...")
    model, _ = train_xgboost(month1[cols], month1["isFraud"], month1[cols])  # dummy test arg, unused output

    results = {}
    for month in range(1, 7):
        month_df = df[df["month"] == month]
        scores = model.predict_proba(month_df[cols])[:, 1]
        y = month_df["isFraud"].to_numpy()
        metrics = evaluate(y, scores)
        metrics["n_rows"] = len(month_df)
        metrics["fraud_rate"] = float(y.mean())
        results[f"month_{month}"] = metrics
        print(f"  month {month}: PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
              f"recall@1%FPR={metrics['recall_at_1pct_fpr']:.4f}  n={len(month_df):,}  "
              f"fraud_rate={y.mean():.4%}")

    pr_aucs = [results[f"month_{m}"]["pr_auc"] for m in range(1, 7)]
    decay = pr_aucs[0] - pr_aucs[-1]
    print(f"\nPR-AUC decay, month 1 -> month 6: {pr_aucs[0]:.4f} -> {pr_aucs[-1]:.4f} "
          f"(delta {decay:+.4f}, {decay / pr_aucs[0]:+.1%} relative)")

    results["summary"] = {
        "month1_pr_auc": pr_aucs[0],
        "month6_pr_auc": pr_aucs[-1],
        "absolute_decay": decay,
        "relative_decay": decay / pr_aucs[0],
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
