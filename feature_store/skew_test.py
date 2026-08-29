"""Training-serving skew test: for a sample of card1 keys, does Feast's online store (Redis,
populated via `feast materialize`) return the same feature values as the latest row in the
offline store (Parquet, the source of truth `materialize` read from)?

This compares Feast's own online vs. offline paths specifically -- not the separate hand-rolled
Redis keys the Flink job writes directly (`velocity:card1:*`, see definitions.py docstring),
which are a different online-serving mechanism entirely and not what Feast's materialize
populates.

Run after `feast materialize` has covered the full replayed time range.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from feast import FeatureStore

DATA_DIR = Path("/home/HP/driftline/data/processed")
WINDOWS = ["1h", "24h", "7d"]


def latest_offline_values(window_label: str) -> pd.DataFrame:
    table = ds.dataset(str(DATA_DIR / f"card_velocity_{window_label}"), format="parquet").to_table()
    df = table.to_pandas()
    # "latest" = the row with the max window_end per card1 -- what materialize should have
    # written to the online store as of the most recent materialization run.
    idx = df.groupby("card1")["window_end"].idxmax()
    return df.loc[idx].set_index("card1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    store = FeatureStore(repo_path=str(Path(__file__).resolve().parent))

    offline_latest = {label: latest_offline_values(label) for label in WINDOWS}
    all_card1s = sorted(set().union(*[set(df.index) for df in offline_latest.values()]))
    sample = all_card1s[: args.sample_size]
    print(f"Comparing {len(sample)} card1 keys across {len(WINDOWS)} feature views...")

    entity_rows = [{"card1": c} for c in sample]
    result = store.get_online_features(
        features=[f"card_velocity_{w}:{f}" for w in WINDOWS for f in ("txn_count", "amt_sum")],
        entity_rows=entity_rows,
        full_feature_names=True,
    ).to_dict()

    max_diff = {"txn_count": 0, "amt_sum": 0.0}
    mismatches = []

    for i, card1 in enumerate(sample):
        for label in WINDOWS:
            online_count = result[f"card_velocity_{label}__txn_count"][i]
            online_amt = result[f"card_velocity_{label}__amt_sum"][i]
            offline_row = offline_latest[label].loc[card1] if card1 in offline_latest[label].index else None

            if offline_row is None:
                continue  # not every card1 has data in every window (sparse traffic)

            offline_count = int(offline_row["txn_count"])
            offline_amt = float(offline_row["amt_sum"])

            if online_count is None:
                mismatches.append((card1, label, "online missing", offline_count, None))
                continue

            count_diff = abs(online_count - offline_count)
            amt_diff = abs(online_amt - offline_amt)
            max_diff["txn_count"] = max(max_diff["txn_count"], count_diff)
            max_diff["amt_sum"] = max(max_diff["amt_sum"], amt_diff)

            if count_diff > 0 or amt_diff > args.tolerance:
                mismatches.append((card1, label, "value mismatch",
                                    (offline_count, offline_amt), (online_count, online_amt)))

    print(f"\nMax abs diff -- txn_count: {max_diff['txn_count']}, amt_sum: {max_diff['amt_sum']}")
    print(f"Mismatches: {len(mismatches)} / {len(sample) * len(WINDOWS)} (card1, window) pairs checked")
    for m in mismatches[:20]:
        print(f"  {m}")

    if mismatches:
        print("\nFAIL -- training-serving skew detected, see mismatches above")
        sys.exit(1)
    print("\nPASS -- online store matches offline latest values within tolerance")


if __name__ == "__main__":
    main()
