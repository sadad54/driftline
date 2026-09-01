"""Answers the 'X flags/day, Y analysts' interview question for real: per-day precision@k under
a fixed daily analyst review budget, using the persisted Phase 1 XGBoost model (not retrained
here -- loaded from models/artifacts/xgboost_baseline.json, produced by scripts/export_onnx.py).
"""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from driftline.data import feature_columns, load_ieee_cis, time_ordered_split

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "artifacts" / "xgboost_baseline.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "analyst_budget_simulation.json"

# analyst budget expressed as a percentage of that day's transaction volume -- mirrors how a
# real fraud team sizes review capacity (proportional to traffic, not a fixed headcount number
# that would be meaningless without knowing typical daily volume)
BUDGET_FRACTIONS = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5% of daily volume


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found -- run scripts/export_onnx.py first")

    df = load_ieee_cis()
    _, test = time_ordered_split(df, test_frac=0.2)
    cols = feature_columns(df)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    scores = model.predict_proba(test[cols])[:, 1]

    test = test.copy()
    test["score"] = scores
    # day bucket: TransactionDT has no calendar meaning (see data/README.md) -- "day" here means
    # a fixed 86400-second bucket from the test split's own start, which is what a real
    # analyst-shift boundary would look like against this data's relative clock.
    dt0 = test["TransactionDT"].min()
    test["day"] = (test["TransactionDT"] - dt0) // 86400

    results = {}
    for frac in BUDGET_FRACTIONS:
        daily_precisions = []
        daily_volumes = []
        for day, group in test.groupby("day"):
            k = max(1, int(len(group) * frac))
            top_k = group.nlargest(k, "score")
            precision = top_k["isFraud"].mean()
            daily_precisions.append(precision)
            daily_volumes.append(len(group))

        results[f"{frac:.1%}_budget"] = {
            "mean_daily_volume": float(np.mean(daily_volumes)),
            "mean_k_per_day": float(np.mean(daily_volumes) * frac),
            "mean_precision_at_k": float(np.mean(daily_precisions)),
            "min_precision_at_k": float(np.min(daily_precisions)),
            "max_precision_at_k": float(np.max(daily_precisions)),
            "n_days": len(daily_precisions),
        }

    print(json.dumps(results, indent=2))
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
