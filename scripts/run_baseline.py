"""Phase 1 deliverable: reproduce the baseline (XGBoost + IsolationForest) on a strict
time-ordered holdout of IEEE-CIS, and record the real baseline PR-AUC / ROC-AUC / recall@1%FPR.

This is the number every later phase (GraphSAGE ensemble ablation, drift-triggered retrain
decay curve) reports its lift against.
"""
import json
import time
from pathlib import Path

from driftline.baseline import evaluate, rank_average, train_isolation_forest, train_xgboost
from driftline.data import feature_columns, load_ieee_cis, time_ordered_split

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "baseline_metrics.json"


def main():
    t0 = time.time()
    print("Loading IEEE-CIS (left join transaction + identity)...")
    df = load_ieee_cis()
    print(f"  loaded {len(df):,} rows, {len(df.columns)} cols in {time.time() - t0:.1f}s")

    train, test = time_ordered_split(df, test_frac=0.2)
    assert train["TransactionDT"].max() <= test["TransactionDT"].min(), "leakage: train overlaps test in time"
    print(f"  train: {len(train):,} rows ({train['TransactionDT'].min()}-{train['TransactionDT'].max()})")
    print(f"  test:  {len(test):,} rows ({test['TransactionDT'].min()}-{test['TransactionDT'].max()})")
    print(f"  train fraud rate: {train['isFraud'].mean():.4%}, test fraud rate: {test['isFraud'].mean():.4%}")

    cols = feature_columns(df)
    X_train, y_train = train[cols], train["isFraud"]
    X_test, y_test = test[cols], test["isFraud"]

    print("\nTraining XGBoost...")
    t0 = time.time()
    xgb_model, xgb_scores = train_xgboost(X_train, y_train, X_test)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training IsolationForest...")
    t0 = time.time()
    iso_model, iso_scores = train_isolation_forest(X_train, X_test)
    print(f"  done in {time.time() - t0:.1f}s")

    combined_scores = rank_average(xgb_scores, iso_scores)

    # analyst review budget: assume review capacity of 1% of daily volume, applied to the whole
    # holdout as a stand-in (Phase 4 will simulate this per-day against a stated daily budget).
    k = max(1, int(0.01 * len(y_test)))

    y_test_np = y_test.to_numpy()
    results = {
        "n_train": len(train),
        "n_test": len(test),
        "train_fraud_rate": float(train["isFraud"].mean()),
        "test_fraud_rate": float(test["isFraud"].mean()),
        "precision_at_k_budget": k,
        "xgboost": evaluate(y_test_np, xgb_scores, k=k),
        "isolation_forest": evaluate(y_test_np, iso_scores, k=k),
        "rank_average_ensemble": evaluate(y_test_np, combined_scores, k=k),
    }

    print("\n=== Baseline results (time-ordered holdout, last 20% by TransactionDT) ===")
    print(json.dumps(results, indent=2))

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
