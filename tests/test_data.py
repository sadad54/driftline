import pandas as pd
import pytest

from driftline.data import add_month_bucket, feature_columns, time_ordered_split


def _toy_df(n=1000, seed=0):
    rng = pd.Series(range(n))
    return pd.DataFrame(
        {
            "TransactionID": rng + 1,
            "TransactionDT": rng * 100 + 86400,
            "isFraud": (rng % 29 == 0).astype(int),
            "TransactionAmt": rng * 1.5,
        }
    )


def test_time_ordered_split_no_leakage():
    """The CI-enforced leakage regression test: max(train time) must never exceed
    min(test time). This is the exact guard the source doc calls for under
    'random train/test split... leaks the future, inflates every number.'"""
    df = _toy_df()
    train, test = time_ordered_split(df, test_frac=0.2)

    assert len(train) + len(test) == len(df)
    assert train["TransactionDT"].max() <= test["TransactionDT"].min()
    # strictly no row of train appears after any row of test chronologically
    assert train["TransactionDT"].max() < test["TransactionDT"].min() or train["TransactionDT"].max() == test["TransactionDT"].min()


def test_time_ordered_split_would_fail_on_shuffled_random_split():
    """Negative control: prove the test above actually catches leakage, by showing a random
    split (the anti-pattern) fails the same assertion."""
    df = _toy_df()
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    bad_train, bad_test = shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]

    assert not (bad_train["TransactionDT"].max() <= bad_test["TransactionDT"].min())


def test_month_bucket_covers_full_range_in_order():
    df = _toy_df()
    bucketed = add_month_bucket(df, n_buckets=6)

    assert bucketed["month"].min() == 1
    assert bucketed["month"].max() == 6
    # bucket assignment must be monotonic non-decreasing with time
    assert (bucketed.sort_values("TransactionDT")["month"].diff().dropna() >= 0).all()


def test_feature_columns_excludes_label_and_id_and_time():
    df = _toy_df()
    df = add_month_bucket(df)
    cols = feature_columns(df)

    assert "isFraud" not in cols
    assert "TransactionID" not in cols
    assert "TransactionDT" not in cols
    assert "month" not in cols
    assert "TransactionAmt" in cols
