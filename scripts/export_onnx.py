"""Train XGBoost on the Phase 1 split, export to ONNX, and verify ONNX-vs-native prediction
parity on the test set.

Real risk being tested here, not assumed: the Phase 1 model was trained with
`enable_categorical=True` (pandas category dtype columns handled natively by XGBoost). ONNX's
tree-ensemble op expects plain numeric input, so this script checks directly whether
onnxmltools' XGBoost converter handles that path or whether a numeric-only "serving" variant is
needed -- and reports whichever is actually true rather than assuming either way.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from driftline.baseline import train_xgboost
from driftline.data import feature_columns, load_ieee_cis, time_ordered_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ieee_cis_ci_sample.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true",
                         help="use the small committed CI sample instead of the full dataset (for CI)")
    args = parser.parse_args()

    print("Loading data and training XGBoost (Phase 1 split)...")
    if args.fixture:
        df = pd.read_parquet(FIXTURE_PATH)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("category")
    else:
        df = load_ieee_cis()
    train, test = time_ordered_split(df, test_frac=0.2)
    cols = feature_columns(df)

    categorical_cols = [c for c in cols if str(train[c].dtype) == "category"]
    print(f"  {len(categorical_cols)} categorical columns (native XGBoost handling): {categorical_cols[:5]}...")

    t0 = time.time()
    model, native_scores = train_xgboost(train[cols], train["isFraud"], test[cols])
    print(f"  trained in {time.time() - t0:.1f}s")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgboost_baseline.json"))

    print("\nAttempting ONNX conversion (categorical dtype path)...")
    try:
        onnx_model = convert_xgboost(
            model.get_booster(),
            initial_types=[("input", FloatTensorType([None, len(cols)]))],
        )
        conversion_ok = True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        conversion_ok = False

    if not conversion_ok:
        print("\nCategorical-native ONNX export failed as suspected -- falling back to a "
              "numeric-only 'serving' variant (ordinal-encode categoricals, no enable_categorical).")
        return train_numeric_serving_variant(train, test, cols)

    onnx_path = MODEL_DIR / "xgboost_baseline.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ONNX model written to {onnx_path}")

    print("\nParity check: ONNX Runtime vs native XGBoost predict_proba...")
    sess = ort.InferenceSession(str(onnx_path))
    X_test = test[cols].astype(np.float32).to_numpy()
    onnx_scores = sess.run(None, {"input": X_test})[1][:, 1]

    max_diff = np.abs(onnx_scores - native_scores).max()
    print(f"  max abs diff: {max_diff}")
    print("PASS" if max_diff < 1e-4 else "FAIL")


def train_numeric_serving_variant(train, test, cols):
    """Serving-model variant: same features, but ordinal-encoded (not pandas category dtype) so
    ONNX export is unambiguous. A real, named difference from the Phase 1 'research' model --
    documented, not silently swapped."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    train_enc = train[cols].copy()
    test_enc = test[cols].copy()
    category_maps = {}  # persisted so the serving service can reproduce this exact encoding
    for col in cols:
        if str(train[col].dtype) == "category":
            cats = train[col].cat.categories
            category_maps[col] = cats.tolist()
            train_enc[col] = train[col].cat.codes.astype("float32")
            test_enc[col] = test[col].astype(pd.CategoricalDtype(categories=cats)).cat.codes.astype("float32")

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=(train["isFraud"] == 0).sum() / (train["isFraud"] == 1).sum(),
        n_jobs=-1, random_state=0,
    )
    # Fit on a plain numpy array, not the DataFrame: fitting on a DataFrame makes XGBoost store
    # the real pandas column names (e.g. 'V258') as its internal feature_names, which
    # onnxmltools' converter can't parse (it expects the default 'f0','f1',... pattern XGBoost
    # uses when it has no column names to go on). Found via the RuntimeError this produced.
    X_train_np = train_enc.fillna(-1).to_numpy(dtype=np.float32)
    X_test_np = test_enc.fillna(-1).to_numpy(dtype=np.float32)
    model.fit(X_train_np, train["isFraud"])
    native_scores = model.predict_proba(X_test_np)[:, 1]

    onnx_model = convert_xgboost(
        model.get_booster(), initial_types=[("input", FloatTensorType([None, len(cols)]))]
    )
    onnx_path = MODEL_DIR / "xgboost_serving.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ONNX serving model written to {onnx_path}")

    preprocessing = {"feature_columns": cols, "category_maps": category_maps}
    with open(MODEL_DIR / "serving_preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)
    print(f"  preprocessing metadata written to {MODEL_DIR / 'serving_preprocessing.json'}")

    sess = ort.InferenceSession(str(onnx_path))
    onnx_scores = sess.run(None, {"input": X_test_np})[1][:, 1]

    max_diff = np.abs(onnx_scores - native_scores).max()
    pr_auc_native = average_precision_score(test["isFraud"], native_scores)
    pr_auc_onnx = average_precision_score(test["isFraud"], onnx_scores)
    print(f"  max abs diff: {max_diff}")
    print(f"  PR-AUC native={pr_auc_native:.4f}, onnx={pr_auc_onnx:.4f}")
    print("PASS" if max_diff < 1e-4 else "FAIL")


if __name__ == "__main__":
    import pandas as pd  # noqa: E402  (only needed in the fallback path)
    main()
