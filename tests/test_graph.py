import numpy as np
import pandas as pd

from driftline.graph import ValueNodeVocab, build_graph_edges


def _toy_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "card1": rng.integers(1, 5, size=n),
        "card2": rng.integers(100, 103, size=n),
        "card3": rng.integers(1, 3, size=n),
        "card5": rng.integers(1, 3, size=n),
        "addr1": rng.integers(1, 4, size=n),
        "addr2": rng.integers(1, 2, size=n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], size=n),
        "R_emaildomain": rng.choice(["gmail.com", None], size=n),
    })


def test_train_only_graph_node_numbering_matches_train_size():
    """Node numbering invariant: building a graph from train_df alone must number its txn nodes
    [0, len(train_df)) and its value nodes [len(train_df), len(train_df) + V) -- never wider than
    that, i.e. structurally incapable of referencing a test-set row as a node."""
    full_df = _toy_df(n=40)
    train_df = full_df.iloc[:25].reset_index(drop=True)

    vocab = ValueNodeVocab().fit(full_df)
    edge_index = build_graph_edges(train_df, vocab)

    max_node_id = edge_index.max().item()
    # every node id used must be < len(train_df) (a txn node) or >= len(train_df) (a value node,
    # offset by build_graph_edges) but the graph was built with num_txn=len(train_df)=25, so no
    # id in [25, ...) can be mistaken for a 26th-40th test-set transaction -- there's no
    # numbering slot for one, by construction.
    assert max_node_id < len(train_df) + len(vocab)


def test_value_only_present_in_test_rows_never_appears_in_train_graph():
    """The leakage property that actually matters: a (column, value) that only occurs in
    test-split rows must never appear as an edge target in a graph built from train_df alone --
    otherwise a value unique to the future would still influence training-graph structure."""
    train_df = pd.DataFrame({
        "card1": [1, 2, 3], "card2": [100, 100, 100], "card3": [1, 1, 1], "card5": [1, 1, 1],
        "addr1": [1, 1, 1], "addr2": [1, 1, 1], "P_emaildomain": ["gmail.com"] * 3,
        "R_emaildomain": [None, None, None],
    })
    test_df = pd.DataFrame({
        "card1": [999], "card2": [100], "card3": [1], "card5": [1],
        "addr1": [1], "addr2": [1], "P_emaildomain": ["gmail.com"], "R_emaildomain": [None],
    })
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    vocab = ValueNodeVocab().fit(full_df)

    future_only_value_id = vocab.lookup("card1", 999)
    assert future_only_value_id is not None

    train_edges = build_graph_edges(train_df, vocab)
    train_value_ids = set((train_edges[1] - len(train_df)).tolist()) | \
        set((train_edges[0] - len(train_df)).tolist())
    assert future_only_value_id not in train_value_ids


def test_edges_are_bidirectional():
    df = _toy_df(n=10)
    vocab = ValueNodeVocab().fit(df)
    edge_index = build_graph_edges(df, vocab)
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    reverse_pairs = set(zip(edge_index[1].tolist(), edge_index[0].tolist()))
    assert pairs == reverse_pairs
