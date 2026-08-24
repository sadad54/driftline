"""Loading and time-ordered splitting for the IEEE-CIS fraud dataset.

Ground truth facts this module encodes (verified via scripts/inspect_ieee_cis.py on the
real downloaded files, not assumed from the source doc):
  - train_transaction.csv: 590,540 rows, 394 cols, isFraud rate 3.499%.
  - TransactionDT is a *relative* offset in seconds (starts at 86400 = day 1), not a real
    calendar timestamp — it spans ~182 days (~6 months). There is no wall-clock date.
  - Only 24.4% of transactions have a matching row in train_identity.csv (left join required,
    NOT inner — an inner join would silently drop 75.6% of the data).
  - test_transaction.csv / test_identity.csv are Kaggle's blind leaderboard set: NO isFraud
    labels. They are unusable for our own evaluation. All train/holdout splits must be carved
    out of train_transaction.csv itself.
  - card4, card6, P_emaildomain, R_emaildomain, DeviceType, DeviceInfo, id_12..id_38 are
    object-dtype categoricals; everything else is numeric. XGBoost 2.x handles both NaN and
    pandas `category` dtype natively (tree_method="hist", enable_categorical=True), so no
    manual imputation/encoding is needed for the XGBoost path.
"""
from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ieee-cis"

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
] + [f"id_{i}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)]


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Halve memory footprint (float64->float32, int64->int32/16) column-by-column, in place.

    Required on this build machine (7.3GB total RAM): the naive merged frame's dense float64
    block alone needs ~1.76GB to consolidate, which OOMs. Column-wise downcast avoids ever
    materializing that block at full precision.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_ieee_cis(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and left-join train_transaction + train_identity on TransactionID.

    Left join (not inner): identity data is only present for 24.4% of transactions, and its
    absence is itself informative (e.g. correlates with ProductCD / channel), not something to
    discard rows over.
    """
    txn = _downcast_numeric(pd.read_csv(raw_dir / "train_transaction.csv", engine="pyarrow"))
    ident = _downcast_numeric(pd.read_csv(raw_dir / "train_identity.csv", engine="pyarrow"))
    df = txn.merge(ident, on="TransactionID", how="left")
    del txn, ident
    gc.collect()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df.sort_values("TransactionDT", inplace=True, ignore_index=True, kind="mergesort")
    return df


def add_month_bucket(df: pd.DataFrame, n_buckets: int = 6) -> pd.DataFrame:
    """Bucket TransactionDT into n_buckets equal-width chronological buckets (1-indexed).

    TransactionDT has no calendar meaning, so "month" here means "1/6th of the observed
    time span", used consistently everywhere the six-month decay-curve story is told
    (Phase 5's PSI/retrain work reuses this exact bucketing).
    """
    dt_min, dt_max = df["TransactionDT"].min(), df["TransactionDT"].max()
    span = dt_max - dt_min
    bucket_width = span / n_buckets
    df = df.copy()
    df["month"] = ((df["TransactionDT"] - dt_min) // bucket_width).clip(upper=n_buckets - 1).astype(int) + 1
    return df


def time_ordered_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last `test_frac` of rows BY TIME become the holdout.

    Never use a random split here — see tests/test_data.py::test_time_ordered_split_no_leakage,
    which is the CI-enforced leakage regression test the source doc calls for.
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train, test


FEATURE_COLS_EXCLUDE = {"TransactionID", "isFraud", "TransactionDT", "month"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
