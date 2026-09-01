"""Generate real sample /score request payloads from the test split, for the Locust load test.
Real transactions, not synthetic data -- exercises the actual categorical-encoding path."""
import json
from pathlib import Path

import pandas as pd

from driftline.data import feature_columns, load_ieee_cis, time_ordered_split

OUT_PATH = Path(__file__).resolve().parent.parent / "serving" / "sample_requests.json"


def main():
    df = load_ieee_cis()
    _, test = time_ordered_split(df, test_frac=0.2)
    cols = feature_columns(df)

    sample = test.sample(n=min(500, len(test)), random_state=0)
    requests = []
    for _, row in sample.iterrows():
        features = {}
        for col in cols:
            val = row[col]
            if str(df[col].dtype) == "category":
                features[col] = None if pd.isna(val) else str(val)
            else:
                features[col] = None if pd.isna(val) else float(val)
        requests.append({"features": features, "card1": int(row["card1"])})

    with open(OUT_PATH, "w") as f:
        json.dump(requests, f)
    print(f"Wrote {len(requests)} sample requests to {OUT_PATH}")


if __name__ == "__main__":
    main()
