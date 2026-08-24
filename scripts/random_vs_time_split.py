"""The 'your metrics lied to you' artifact: same XGBoost config, same data, evaluated on a
random 80/20 split vs. the honest time-ordered split. Quantifies exactly how much a random
split inflates the number on this dataset, instead of asserting it abstractly.
"""
import gc
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from driftline.baseline import evaluate, train_xgboost
from driftline.data import feature_columns, load_ieee_cis, time_ordered_split

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "random_vs_time_split.json"


def main():
    df = load_ieee_cis()
    cols = feature_columns(df)

    # Time-ordered (honest) — reuse the same split as the baseline run.
    train_t, test_t = time_ordered_split(df, test_frac=0.2)
    y_test_t = test_t["isFraud"].to_numpy()
    model_t, scores_time = train_xgboost(train_t[cols], train_t["isFraud"], test_t[cols])
    time_metrics = evaluate(y_test_t, scores_time)

    # Free the time-split model/frames before building the random split — this machine has
    # only 7.3GB RAM and holding df + train_t + test_t + train_r + test_r simultaneously OOMs.
    del train_t, test_t, model_t, scores_time
    gc.collect()

    # Random 80/20 (the anti-pattern) — same model config, same overall data, only the split changes.
    train_r, test_r = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["isFraud"]
    )
    del df
    gc.collect()
    y_test_r = test_r["isFraud"].to_numpy()
    _, scores_random = train_xgboost(train_r[cols], train_r["isFraud"], test_r[cols])
    random_metrics = evaluate(y_test_r, scores_random)

    result = {
        "time_ordered_split": time_metrics,
        "random_split": random_metrics,
        "pr_auc_inflation": random_metrics["pr_auc"] - time_metrics["pr_auc"],
        "roc_auc_inflation": random_metrics["roc_auc"] - time_metrics["roc_auc"],
        "explanation": (
            "A random split lets rows from the same card/device/session appear in both train "
            "and test (IEEE-CIS transactions are not i.i.d. across time — repeated cards, "
            "addresses, and email domains recur within short windows). The model partially "
            "memorizes entity identity rather than learning genuinely predictive fraud "
            "patterns, so the random-split holdout score is inflated relative to the honest "
            "forward-looking time-ordered evaluation, which is what production actually faces."
        ),
    }
    print(json.dumps(result, indent=2))
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
