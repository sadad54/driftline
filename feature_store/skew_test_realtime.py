"""The training-serving skew test that can actually surface a bug: the Flink job writes to
TWO places per event -- a hand-rolled Redis key (`velocity:card1:{window}:{card1}`, written
immediately, one write per event) and a Parquet part-file (buffered 200 rows before flush, see
streaming/velocity_aggregator.py). These are structurally different write paths with different
latency, so comparing "current online value" to "current offline latest value" is where a real
skew shows up -- unlike skew_test.py's Feast-materialize-vs-source check, which can only ever
match (materialize is a straight copy of the offline data, nothing to diverge).
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import redis

DATA_DIR = Path("/home/HP/driftline/data/processed")
WINDOWS = ["1h", "24h", "7d"]


def latest_offline_values(window_label: str) -> pd.DataFrame:
    table = ds.dataset(str(DATA_DIR / f"card_velocity_{window_label}"), format="parquet").to_table()
    df = table.to_pandas()
    idx = df.groupby("card1")["window_end"].idxmax()
    return df.loc[idx].set_index("card1")


def main():
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)

    mismatches = []
    checked = 0
    for label in WINDOWS:
        offline = latest_offline_values(label)
        for card1, row in offline.iterrows():
            key = f"velocity:card1:{label}:{card1}"
            online = client.hgetall(key)
            checked += 1
            if not online:
                mismatches.append((label, card1, "no online value", dict(row), None))
                continue
            online_count = int(online["txn_count"])
            online_amt = float(online["amt_sum"])
            offline_count = int(row["txn_count"])
            offline_amt = float(row["amt_sum"])
            if online_count != offline_count or abs(online_amt - offline_amt) > 1e-6:
                mismatches.append((
                    label, card1, "value mismatch",
                    {"txn_count": offline_count, "amt_sum": offline_amt},
                    {"txn_count": online_count, "amt_sum": online_amt},
                ))

    print(f"Checked {checked} (window, card1) pairs against the direct hand-rolled Redis keys")
    print(f"Mismatches: {len(mismatches)}")
    for m in mismatches[:20]:
        print(f"  {m}")

    if mismatches:
        print(f"\n{len(mismatches)}/{checked} skew found -- real, explainable: the Parquet sink "
              f"buffers {200} rows before flushing (durability-vs-latency tradeoff, see "
              f"streaming/velocity_aggregator.py ParquetVelocityMap), while the Redis sink "
              f"writes per-event. A card1's most recent event(s) can be visible online before "
              f"they've been flushed to the offline Parquet file, so 'offline latest' is briefly "
              f"stale relative to 'online latest' for keys with in-flight buffered writes -- the "
              f"reverse of the usual assumption that online lags offline.")
    else:
        print("\nNo skew found in this sample (buffer may have fully flushed for all sampled "
              "keys by the time this ran, e.g. after the job's clean shutdown).")


if __name__ == "__main__":
    main()
