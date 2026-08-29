"""Feast feature definitions for Driftline's card-velocity features.

Entity: card1 (the card identifier IEEE-CIS transactions are keyed by). Feature views wrap the
Parquet part-file directories the PyFlink job (streaming/velocity_aggregator.py) writes as its
offline sink; Feast's online store (Redis) is populated by `feast materialize`, which is
DISTINCT from the Redis keys the Flink job itself writes directly (`velocity:card1:*`) --
those are a hand-rolled online store for the pipeline's own low-latency lookups, while this
Feast-managed online store is what the Phase 4 scoring service will actually query through the
Feast SDK. Both point at genuinely computed features; documenting the two paths so it's not
mistaken for one thing.

Path note: FileSource paths are absolute (this VM's home directory) rather than relative --
simplest, least ambiguous option for a single-machine demo repo; would need to become
parameterized (env var or repo-relative) for a multi-environment deployment.
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

card1 = Entity(name="card1", join_keys=["card1"])

DATA_DIR = "/home/HP/driftline/data/processed"

card_velocity_1h_source = FileSource(
    name="card_velocity_1h_source",
    path=f"{DATA_DIR}/card_velocity_1h",
    timestamp_field="window_end",
)
card_velocity_24h_source = FileSource(
    name="card_velocity_24h_source",
    path=f"{DATA_DIR}/card_velocity_24h",
    timestamp_field="window_end",
)
card_velocity_7d_source = FileSource(
    name="card_velocity_7d_source",
    path=f"{DATA_DIR}/card_velocity_7d",
    timestamp_field="window_end",
)

card_velocity_1h_fv = FeatureView(
    name="card_velocity_1h",
    entities=[card1],
    ttl=timedelta(days=400),  # covers the full ~182-day replay span with headroom
    schema=[
        Field(name="txn_count", dtype=Int64),
        Field(name="amt_sum", dtype=Float64),
    ],
    source=card_velocity_1h_source,
    online=True,
)
card_velocity_24h_fv = FeatureView(
    name="card_velocity_24h",
    entities=[card1],
    ttl=timedelta(days=400),
    schema=[
        Field(name="txn_count", dtype=Int64),
        Field(name="amt_sum", dtype=Float64),
    ],
    source=card_velocity_24h_source,
    online=True,
)
card_velocity_7d_fv = FeatureView(
    name="card_velocity_7d",
    entities=[card1],
    ttl=timedelta(days=400),
    schema=[
        Field(name="txn_count", dtype=Int64),
        Field(name="amt_sum", dtype=Float64),
    ],
    source=card_velocity_7d_source,
    online=True,
)
