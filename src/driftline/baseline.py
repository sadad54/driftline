"""XGBoost + IsolationForest baseline — Phase 1 reference point.

This reproduces (on the new time-ordered split) the same model family the user's prior fraud
project used, but honestly: time-ordered holdout, PR-AUC as the headline metric (not ROC-AUC),
and recall at a fixed low false-positive rate, which is what a real fraud team budgets analyst
review capacity against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    top_k = y_true[order[:k]]
    return float(top_k.mean())


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """XGBoost with native NaN + pandas-category handling — no imputation, no manual encoding."""
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        n_jobs=-1,
        random_state=0,
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    return model, scores


def train_isolation_forest(X_train: pd.DataFrame, X_test: pd.DataFrame) -> np.ndarray:
    """Unsupervised anomaly signal on numeric-only, median-imputed features.

    IsolationForest can't consume NaN or pandas categoricals, unlike the XGBoost path — so this
    is a genuinely separate feature matrix, not a shortcut. Categoricals are ordinal-coded
    (their category codes), which is semantically weak but standard for this kind of auxiliary
    anomaly signal and is called out as such rather than hidden.
    """
    numeric_train = X_train.copy()
    numeric_test = X_test.copy()
    for col in numeric_train.columns:
        if str(numeric_train[col].dtype) == "category":
            codes_train = numeric_train[col].cat.codes
            numeric_train[col] = codes_train.replace(-1, np.nan)
            numeric_test[col] = numeric_test[col].astype(
                pd.CategoricalDtype(categories=X_train[col].cat.categories)
            ).cat.codes.replace(-1, np.nan)

    medians = numeric_train.median(numeric_only=True)
    numeric_train = numeric_train.fillna(medians)
    numeric_test = numeric_test.fillna(medians)

    model = IsolationForest(n_estimators=200, contamination="auto", n_jobs=-1, random_state=0)
    model.fit(numeric_train)
    # score_samples: higher = more normal. Flip sign so higher = more anomalous = more fraud-like.
    scores = -model.score_samples(numeric_test)
    return model, scores


def rank_average(*score_arrays: np.ndarray) -> np.ndarray:
    ranks = [pd.Series(s).rank(pct=True).to_numpy() for s in score_arrays]
    return np.mean(ranks, axis=0)


def evaluate(y_true: np.ndarray, scores: np.ndarray, k: int | None = None) -> dict:
    result = {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
    }
    if k is not None:
        result["precision_at_k"] = precision_at_k(y_true, scores, k)
    return result
