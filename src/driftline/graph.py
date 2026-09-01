"""Entity graph construction for the GraphSAGE fraud model.

Design: a heterogeneous "shared identity" graph, represented as one homogeneous PyG graph with
two kinds of nodes sharing an embedding space:
  - TRANSACTION nodes: one per row, real numeric features (TransactionAmt, C/D/V columns etc.).
  - VALUE nodes: one per distinct (column, value) pair across card1, card2, card3, card5, addr1,
    addr2, P_emaildomain, R_emaildomain. These carry no "real" features of their own (they're
    anonymous identity anchors) -- the model gives them a learned nn.Embedding instead.
An edge connects a transaction to every value node it has a non-null value for. This is edges
LINEAR in (rows x columns), not O(rows^2) -- avoids the combinatorial blow-up of directly
connecting every pair of transactions that share a card/email/etc.

Why NOT a full "connect every pair of transactions sharing a value" graph: a popular card1 value
used by thousands of transactions would create a near-complete subgraph for that card alone --
infeasible at this scale. Message passing through a shared value node achieves the same
"transactions with a shared identity influence each other's embedding" effect, at edge count
O(rows x id_columns) instead of O(rows^2).

Temporal leakage boundary: THE PROPERTY THAT ACTUALLY MATTERS is that the TRAINING graph is
built from train-split rows only -- GraphSAGE's weights are learned exclusively from
neighborhoods that can never contain a test-set transaction, which is the real leakage risk (a
model's learned weights implicitly encoding future information). This module builds a graph from
whatever dataframe it's given; scripts/train_graphsage.py enforces the boundary by building one
graph from `train` alone for training, and only extending it with `test` rows for read-only
inference afterward (frozen weights). Documented explicitly, not left implicit: within the
extended train+test graph used for inference, test transactions CAN see each other through a
shared value node (a bounded, named simplification -- true per-transaction sequential temporal
masking would require O(len(test)) incremental graph rebuilds, judged not worth the build-time
cost here). See tests/test_graph.py for the regression test on the boundary that matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

ID_COLUMNS = ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]


class ValueNodeVocab:
    """Maps (column, value) pairs to a stable global value-node id, fit once on the union of
    train+test so ids are consistent across the train-only and train+test graphs."""

    def __init__(self):
        self.mapping: dict[tuple[str, object], int] = {}

    def fit(self, df: pd.DataFrame, id_columns: list[str] = ID_COLUMNS) -> "ValueNodeVocab":
        next_id = 0
        for col in id_columns:
            for val in df[col].dropna().unique():
                key = (col, val)
                if key not in self.mapping:
                    self.mapping[key] = next_id
                    next_id += 1
        return self

    def __len__(self):
        return len(self.mapping)

    def lookup(self, col: str, val) -> int | None:
        return self.mapping.get((col, val))


def build_graph_edges(df: pd.DataFrame, vocab: ValueNodeVocab, id_columns: list[str] = ID_COLUMNS):
    """Build a bidirectional edge_index for the transactions in `df` (positionally indexed 0..n-1
    as transaction node ids) against the given (pre-fit) value-node vocabulary.

    Returns edge_index as a [2, E] int64 tensor (already made undirected -- both directions
    present, which is what PyG's SAGEConv expects for symmetric neighbor aggregation).
    """
    src, dst = [], []
    for col in id_columns:
        col_vals = df[col].to_numpy()
        for txn_idx, val in enumerate(col_vals):
            if pd.isna(val):
                continue
            value_node_id = vocab.lookup(col, val)
            if value_node_id is None:
                continue
            src.append(txn_idx)
            dst.append(value_node_id)

    txn_to_value = torch.tensor([src, dst], dtype=torch.long)
    value_to_txn = torch.tensor([dst, src], dtype=torch.long)
    edge_index = torch.cat([txn_to_value, value_to_txn], dim=1)
    return edge_index


NUMERIC_FEATURE_PREFIXES = ("TransactionAmt", "C", "D", "V")


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Same spirit as the IsolationForest numeric matrix in baseline.py: a real, if imperfect,
    numeric feature set for transaction nodes. Restricted to TransactionAmt + C*/D*/V* columns
    (the IEEE-CIS "engineered count/timedelta/Vesta" feature families) -- excludes id/categorical
    columns entirely rather than ordinal-encoding them here, since those are exactly what the
    graph structure itself already represents relationally.
    """
    cols = [c for c in df.columns if c.startswith(NUMERIC_FEATURE_PREFIXES)]
    return [c for c in cols if df[c].dtype.kind in "fi"]


def build_numeric_features(df: pd.DataFrame, feature_cols: list[str], mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Median-impute (using the passed-in, train-fit mean/std) then z-score standardize.

    mean/std MUST be fit on the training split only and reused as-is for test/inference --
    fitting them on test data would itself be a (mild) leakage bug of exactly the kind this
    project's whole point is to avoid.
    """
    X = df[feature_cols].to_numpy(dtype=np.float64)
    nan_mask = np.isnan(X)
    X = np.where(nan_mask, mean, X)
    X = (X - mean) / std
    return torch.tensor(X, dtype=torch.float32)
