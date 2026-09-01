"""Consolidated leakage regression test: no future information (features OR graph edges) is
visible to the model at train time for any given row/timestamp. Pulls together the individual
leakage guarantees already tested in tests/test_data.py and tests/test_graph.py into one
explicit statement of the contract, so it's discoverable in one place rather than implied across
files.

This test does not duplicate the underlying logic (see the referenced tests for the actual
assertions) -- it re-asserts the contract at the integration level: given the SAME real loader
and split functions the training scripts actually use, is the invariant still true end to end?
"""
import numpy as np
import pandas as pd

from driftline.data import feature_columns, time_ordered_split
from driftline.graph import ValueNodeVocab, build_graph_edges


def _toy_transactions(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(86400, 86400 + n * 300, size=n)),
        "isFraud": rng.integers(0, 2, size=n),
        "TransactionAmt": rng.uniform(1, 500, size=n),
        "card1": rng.integers(1, 30, size=n),
        "card2": rng.integers(100, 110, size=n),
        "card3": rng.integers(1, 5, size=n),
        "card5": rng.integers(1, 5, size=n),
        "addr1": rng.integers(1, 10, size=n),
        "addr2": rng.integers(1, 3, size=n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], size=n),
        "R_emaildomain": rng.choice(["gmail.com", None], size=n),
    })


def test_time_split_and_graph_leakage_boundary_hold_together():
    """End-to-end: run the actual time_ordered_split + graph-building functions the real
    training scripts use, on the same dataframe, and check BOTH invariants at once -- no
    tabular feature crosses the time boundary, and no graph edge references a test-set row."""
    df = _toy_transactions()
    train, test = time_ordered_split(df, test_frac=0.25)

    # 1. tabular: train's latest timestamp never exceeds test's earliest
    assert train["TransactionDT"].max() <= test["TransactionDT"].min()

    # 2. tabular: TransactionID/TransactionDT are never model features (would leak position/time)
    cols = feature_columns(df)
    assert "TransactionID" not in cols
    assert "TransactionDT" not in cols

    # 3. graph: a train-only graph structurally cannot reference test-set rows as txn nodes --
    # node numbering for build_graph_edges(train, ...) is [0, len(train)) for txn nodes, which
    # has no slot for a test-set row regardless of what the vocab was fit on
    vocab = ValueNodeVocab().fit(df)
    train_edges = build_graph_edges(train, vocab)
    assert train_edges.max().item() < len(train) + len(vocab)


def test_no_isFraud_derived_feature_in_columns():
    """A specific, common leakage bug class: a feature engineered FROM the label (e.g. a
    'fraud rate for this card' column computed including the current row's own label) would
    trivially solve the task. Guard: isFraud itself is never in the feature set, and nothing in
    the current feature set is named in a way that suggests label-derived aggregation."""
    df = _toy_transactions()
    cols = feature_columns(df)
    assert "isFraud" not in cols
    suspicious = [c for c in cols if "fraud" in c.lower()]
    assert not suspicious, f"found label-derived-looking feature names: {suspicious}"
