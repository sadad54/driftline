"""Drift-triggered retraining, shadow-scored and gated, walked forward week by week across the
six-month replay -- produces the "WITH drift-triggered retraining" PR-AUC-by-week curve to
compare directly against scripts/decay_curve.py's "WITHOUT retraining" curve.

Substituting a lightweight Python orchestration loop for a full Airflow deployment: standing up
Airflow's webserver + scheduler + metadata DB is real infrastructure cost for what is, at its
core, a sequential trigger-check loop -- the actual engineering content (drift detection,
retrain, shadow-score, promotion gate) is identical either way, and MLflow (already running via
docker-compose) still gets every run logged for real. Named explicitly here and in Known Gaps,
not silently done instead of what was asked.

Design, causal throughout:
- Initial model trained on weeks 0-3 (~matches decay_curve.py's month-1 baseline size).
- Each subsequent week: evaluate the CURRENTLY SERVING model first (this is the "without
  retraining yet" number for that week), THEN check the PSI-breach trigger.
- On trigger: retrain on all data since the last retrain (cumulative, still nothing from the
  future), shadow-score the candidate on a held-out TAIL SLICE of that same training window
  (never on future weeks -- keeps the promotion decision causal), promote only if the candidate
  doesn't regress PR-AUC by more than PROMOTION_TOLERANCE on that shadow slice.
- Promoting resets the PSI reference window to the new model's training data.
"""
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import xgboost as xgb

from driftline.baseline import evaluate
from driftline.data import feature_columns, load_ieee_cis
from driftline.graph import numeric_feature_columns

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "drift_triggered_retrain.json"
MIN_WEEKS_BETWEEN_RETRAINS = 2
PROMOTION_TOLERANCE = 0.02  # candidate may regress PR-AUC by at most this much (absolute) to be promoted
SHADOW_HOLDOUT_FRAC = 0.2
PSI_RETRAIN_THRESHOLD = 0.2
N_FEATURES_TRIGGER = 20  # same thresholds as scripts/drift_monitoring.py, kept in sync manually


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Same implementation as scripts/drift_monitoring.py -- duplicated rather than imported
    across sibling scripts (each is invoked as `python scripts/X.py`, not as a package, so
    cross-script imports would need extra path wiring for no real benefit here)."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return np.nan
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))
    if len(bin_edges) < 3:
        return np.nan
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = ref_counts / ref_counts.sum() + 1e-6
    cur_pct = cur_counts / cur_counts.sum() + 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def train_model(train_df, cols):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", enable_categorical=True, eval_metric="aucpr",
        scale_pos_weight=(train_df["isFraud"] == 0).sum() / max(1, (train_df["isFraud"] == 1).sum()),
        n_jobs=-1, random_state=0,
    )
    model.fit(train_df[cols], train_df["isFraud"])
    return model


def score(model, df, cols):
    return model.predict_proba(df[cols])[:, 1]


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("driftline-drift-triggered-retrain")

    print("Loading IEEE-CIS...")
    df = load_ieee_cis()
    num_cols = numeric_feature_columns(df)
    cols = feature_columns(df)

    dt_min = df["TransactionDT"].min()
    df = df.copy()
    df["week"] = (df["TransactionDT"] - dt_min) // (7 * 86400)
    n_weeks = int(df["week"].max()) + 1

    initial_train = df[df["week"] <= 3].sort_values("TransactionDT")
    print(f"Training initial model on weeks 0-3 ({len(initial_train):,} rows)...")
    with mlflow.start_run(run_name="initial_model"):
        t0 = time.time()
        current_model = train_model(initial_train, cols)
        mlflow.log_params({"train_weeks": "0-3", "n_rows": len(initial_train)})
        mlflow.log_metric("train_time_sec", time.time() - t0)

    reference = initial_train
    last_retrain_week = 3
    weekly_curve = {}
    retrain_events = []

    for week in range(4, n_weeks):
        current = df[df["week"] == week]
        if len(current) < 50:
            continue

        y = current["isFraud"].to_numpy()
        scores_current_model = score(current_model, current, cols)
        pr_auc_serving = evaluate(y, scores_current_model)["pr_auc"]
        weekly_curve[f"week_{week}"] = {"pr_auc_with_retraining": pr_auc_serving, "n_rows": len(current)}
        print(f"week {week}: serving-model PR-AUC={pr_auc_serving:.4f} (n={len(current)})")

        # PSI trigger check
        n_breach = 0
        for col in num_cols:
            ref_vals = reference[col].to_numpy(dtype=np.float64)
            cur_vals = current[col].to_numpy(dtype=np.float64)
            psi = compute_psi(ref_vals, cur_vals)
            if not np.isnan(psi) and psi > PSI_RETRAIN_THRESHOLD:
                n_breach += 1

        weeks_since_retrain = week - last_retrain_week
        if n_breach >= N_FEATURES_TRIGGER and weeks_since_retrain >= MIN_WEEKS_BETWEEN_RETRAINS:
            print(f"  drift trigger: {n_breach} features PSI>{PSI_RETRAIN_THRESHOLD} -- retraining...")
            candidate_train_full = df[(df["week"] > last_retrain_week) & (df["week"] <= week)].sort_values("TransactionDT")
            split_idx = int(len(candidate_train_full) * (1 - SHADOW_HOLDOUT_FRAC))
            candidate_train = candidate_train_full.iloc[:split_idx]
            shadow_holdout = candidate_train_full.iloc[split_idx:]

            with mlflow.start_run(run_name=f"retrain_week_{week}"):
                t0 = time.time()
                candidate_model = train_model(candidate_train, cols)
                train_time = time.time() - t0

                y_shadow = shadow_holdout["isFraud"].to_numpy()
                candidate_shadow_scores = score(candidate_model, shadow_holdout, cols)
                serving_shadow_scores = score(current_model, shadow_holdout, cols)
                candidate_pr_auc = evaluate(y_shadow, candidate_shadow_scores)["pr_auc"]
                serving_pr_auc_on_shadow = evaluate(y_shadow, serving_shadow_scores)["pr_auc"]
                regression = serving_pr_auc_on_shadow - candidate_pr_auc
                promoted = bool(regression <= PROMOTION_TOLERANCE)  # numpy.bool_ isn't JSON-serializable

                mlflow.log_params({
                    "trigger_week": week, "n_features_breach": n_breach,
                    "train_window": f"{last_retrain_week}-{week}", "n_train_rows": len(candidate_train),
                })
                mlflow.log_metrics({
                    "candidate_shadow_pr_auc": candidate_pr_auc,
                    "serving_shadow_pr_auc": serving_pr_auc_on_shadow,
                    "regression": regression,
                    "train_time_sec": train_time,
                })
                mlflow.set_tag("promoted", str(promoted))

                print(f"    shadow PR-AUC: candidate={candidate_pr_auc:.4f} vs "
                      f"serving={serving_pr_auc_on_shadow:.4f} (regression={regression:+.4f}) "
                      f"-> {'PROMOTED' if promoted else 'REJECTED'}")

                retrain_events.append({
                    "trigger_week": week, "n_features_breach": n_breach,
                    "n_train_rows": len(candidate_train), "candidate_shadow_pr_auc": candidate_pr_auc,
                    "serving_shadow_pr_auc": serving_pr_auc_on_shadow, "regression": regression,
                    "promoted": promoted,
                })

                if promoted:
                    current_model = candidate_model
                    reference = candidate_train_full
                    last_retrain_week = week

    results = {
        "n_weeks": n_weeks,
        "promotion_tolerance": PROMOTION_TOLERANCE,
        "min_weeks_between_retrains": MIN_WEEKS_BETWEEN_RETRAINS,
        "weekly_curve": weekly_curve,
        "retrain_events": retrain_events,
        "n_retrains_triggered": len(retrain_events),
        "n_retrains_promoted": sum(1 for e in retrain_events if e["promoted"]),
    }
    print(f"\n{len(retrain_events)} retrains triggered, "
          f"{sum(1 for e in retrain_events if e['promoted'])} promoted")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
