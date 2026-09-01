"""Phase 3: train GraphSAGE on the entity graph, ensemble with the Phase 1 XGBoost baseline,
produce the ablation table and the no-history-slice lift analysis the source doc calls for.

CPU-only (this VM has no GPU) -- hyperparameters (hidden_dim, epochs, neighbor fanout) are kept
conservative on purpose and the real wall-clock cost is reported rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from driftline.baseline import evaluate, rank_average, train_isolation_forest, train_xgboost
from driftline.data import feature_columns, load_ieee_cis, time_ordered_split
from driftline.graph import (
    ID_COLUMNS,
    ValueNodeVocab,
    build_graph_edges,
    build_numeric_features,
    numeric_feature_columns,
)
from driftline.graphsage_model import EntitySAGE

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "graphsage_results.json"

HIDDEN_DIM = 64
EPOCHS = 5
NEIGHBOR_FANOUT = [15, 10]
BATCH_SIZE = 1024
LR = 0.005


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="truncate to first N rows for a fast smoke test")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading IEEE-CIS...")
    df = load_ieee_cis()
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    print(f"  loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    train, test = time_ordered_split(df, test_frac=0.2)
    n_train, n_test = len(train), len(test)
    print(f"  train: {n_train:,}, test: {n_test:,}")

    # ---- entity graph ----
    print("\nBuilding entity graph vocab (fit on full df)...")
    t0 = time.time()
    vocab = ValueNodeVocab().fit(df, ID_COLUMNS)
    print(f"  {len(vocab):,} distinct identity values across {ID_COLUMNS} in {time.time() - t0:.1f}s")

    print("Building train-only graph edges...")
    t0 = time.time()
    train_edges = build_graph_edges(train, vocab, ID_COLUMNS)
    print(f"  {train_edges.shape[1]:,} directed edges in {time.time() - t0:.1f}s")

    print("Building train+test graph edges (inference-time extension)...")
    t0 = time.time()
    full_concat = pd.concat([train, test], ignore_index=True)
    full_edges = build_graph_edges(full_concat, vocab, ID_COLUMNS)
    print(f"  {full_edges.shape[1]:,} directed edges in {time.time() - t0:.1f}s")

    # ---- numeric features (fit on train only) ----
    num_cols = numeric_feature_columns(df)
    print(f"\n{len(num_cols)} numeric feature columns for graph nodes")
    train_numeric_raw = train[num_cols].to_numpy(dtype=np.float64)
    mean = np.nanmedian(train_numeric_raw, axis=0)
    std = np.nanstd(train_numeric_raw, axis=0)
    std[std == 0] = 1.0

    train_features = build_numeric_features(train, num_cols, mean, std)
    test_features = build_numeric_features(test, num_cols, mean, std)
    full_features = torch.cat([train_features, test_features], dim=0)

    train_y = torch.tensor(train["isFraud"].to_numpy(), dtype=torch.float32)

    # ---- model ----
    model = EntitySAGE(num_numeric_features=len(num_cols), num_value_nodes=len(vocab), hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor((train_y == 0).sum() / (train_y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_graph = Data(edge_index=train_edges, num_nodes=n_train + len(vocab))
    train_loader = NeighborLoader(
        train_graph,
        num_neighbors=NEIGHBOR_FANOUT,
        input_nodes=torch.arange(n_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(f"\nTraining GraphSAGE ({EPOCHS} epoch(s), hidden_dim={HIDDEN_DIM}, fanout={NEIGHBOR_FANOUT})...")
    t0 = time.time()
    model.train()
    n_batches = 0
    total_loss = 0.0
    for epoch in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.n_id, batch.edge_index, train_features, n_train, batch.batch_size)
            y = train_y[batch.n_id[:batch.batch_size]]
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch} batch {n_batches}: avg_loss={total_loss / n_batches:.4f} "
                      f"({elapsed:.1f}s elapsed, {n_batches / elapsed:.2f} batches/sec)")
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s, {n_batches} batches, avg_loss={total_loss / n_batches:.4f}")

    # ---- inference on test (frozen weights, extended train+test graph) ----
    print("\nRunning GraphSAGE inference on test set...")
    t0 = time.time()
    full_graph = Data(edge_index=full_edges, num_nodes=n_train + n_test + len(vocab))
    test_loader = NeighborLoader(
        full_graph,
        num_neighbors=NEIGHBOR_FANOUT,
        input_nodes=torch.arange(n_train, n_train + n_test),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
    )
    model.eval()
    graphsage_scores = np.zeros(n_test)
    filled = np.zeros(n_test, dtype=bool)
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch.n_id, batch.edge_index, full_features, n_train + n_test, batch.batch_size)
            probs = torch.sigmoid(logits).numpy()
            seed_global_ids = batch.n_id[: batch.batch_size].numpy()
            test_positions = seed_global_ids - n_train  # position within the test split
            graphsage_scores[test_positions] = probs
            filled[test_positions] = True
    assert filled.all(), "not every test node was scored -- NeighborLoader input_nodes coverage bug"
    inference_time = time.time() - t0
    print(f"  inference done in {inference_time:.1f}s")

    # ---- XGBoost baseline (same split, for the ensemble) ----
    print("\nTraining XGBoost baseline (same split, for ensemble)...")
    t0 = time.time()
    cols = feature_columns(df)
    _xgb_model, xgb_scores = train_xgboost(train[cols], train["isFraud"], test[cols])
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training IsolationForest (for the full ablation row)...")
    t0 = time.time()
    _, iso_scores = train_isolation_forest(train[cols], test[cols])
    print(f"  done in {time.time() - t0:.1f}s")

    y_test = test["isFraud"].to_numpy()
    k = max(1, int(0.01 * n_test))

    ablation = {
        "xgboost_alone": evaluate(y_test, xgb_scores, k=k),
        "graphsage_alone": evaluate(y_test, graphsage_scores, k=k),
        "xgboost_plus_graphsage": evaluate(y_test, rank_average(xgb_scores, graphsage_scores), k=k),
        "xgboost_plus_graphsage_plus_isoforest": evaluate(
            y_test, rank_average(xgb_scores, graphsage_scores, iso_scores), k=k
        ),
    }

    # ---- no-history ("new card") slice ----
    print("\nComputing no-history (new card1) slice...")
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    df_sorted["card1_prior_count"] = df_sorted.groupby("card1").cumcount()
    # realign to the test split's row order (time_ordered_split sorts the same way)
    test_prior_count = df_sorted["card1_prior_count"].to_numpy()[n_train:n_train + n_test]
    new_card_mask = test_prior_count == 0
    print(f"  {new_card_mask.sum():,} / {n_test:,} test transactions ({new_card_mask.mean():.2%}) "
          f"are a card1's first-ever appearance")

    slice_results = {}
    if new_card_mask.sum() > 20:  # only report if the slice is large enough to be meaningful
        y_slice = y_test[new_card_mask]
        slice_results = {
            "n_new_card": int(new_card_mask.sum()),
            "fraud_rate_new_card": float(y_slice.mean()),
            "xgboost_alone_new_card": evaluate(y_slice, xgb_scores[new_card_mask]),
            "ensemble_new_card": evaluate(
                y_slice, rank_average(xgb_scores, graphsage_scores)[new_card_mask]
            ),
        }

    results = {
        "n_train": n_train,
        "n_test": n_test,
        "num_value_nodes": len(vocab),
        "num_numeric_features": len(num_cols),
        "graphsage_train_time_sec": train_time,
        "graphsage_inference_time_sec": inference_time,
        "graphsage_n_batches": n_batches,
        "graphsage_final_avg_loss": total_loss / n_batches,
        "ablation": ablation,
        "no_history_slice": slice_results,
    }
    print("\n=== Results ===")
    print(json.dumps(results, indent=2))
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
