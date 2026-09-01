import pandas as pd
import torch

from driftline.graph import ID_COLUMNS, ValueNodeVocab, build_graph_edges


def _toy_df(n=20, seed=0):
    import numpy as np
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


def test_train_only_graph_never_references_test_rows():
    """The leakage property that actually matters: a graph built from train_df alone must never
    contain a transaction node beyond len(train_df) -- i.e. test rows are structurally absent,
    not just unlabeled. Training GraphSAGE on this graph is therefore leakage-safe by
    construction, not by hoping the loss function ignores test rows."""
    full_df = _toy_df(n=40)
    train_df = full_df.iloc[:25].reset_index(drop=True)

    vocab = ValueNodeVocab().fit(full_df)  # fit on full vocab (ids just need to be consistent)
    edge_index = build_graph_edges(train_df, vocab)

    txn_node_ids_referenced = edge_index[0][edge_index[0] < len(full_df)]
    # every txn-side node id used in the train-only graph must be < len(train_df), never in
    # [len(train_df), len(full_df)) which would mean a test-set row leaked in as a node
    assert txn_node_ids_referenced.max().item() < len(train_df)


def test_value_node_ids_consistent_across_train_and_full_vocab():
    """The vocab must be fit once (on train+test) and reused, so a value node id means the same
    thing in the train-only graph and the train+test inference-time graph -- otherwise the
    model's learned embedding for a value node wouldn't transfer to inference at all."""
    full_df = _toy_df(n=40)
    train_df = full_df.iloc[:25].reset_index(drop=True)

    vocab = ValueNodeVocab().fit(full_df)
    edges_train = build_graph_edges(train_df, vocab)
    edges_full = build_graph_edges(full_df, vocab)

    # every value-node id that appears in the train-only graph must also appear in the full graph
    # (same vocab), and never exceed len(vocab)
    value_ids_train = set(edges_train[1].tolist()) - set(range(len(train_df)))
    assert all(v < len(vocab) for v in value_ids_train)
    value_ids_full = set(edges_full[1].tolist())
    assert value_ids_train.issubset(value_ids_full | set(range(len(full_df))))


def test_edges_are_bidirectional():
    df = _toy_df(n=10)
    vocab = ValueNodeVocab().fit(df)
    edge_index = build_graph_edges(df, vocab)
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    reverse_pairs = set(zip(edge_index[1].tolist(), edge_index[0].tolist()))
    assert pairs == reverse_pairs
