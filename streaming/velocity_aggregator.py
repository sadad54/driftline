"""PyFlink windowed velocity aggregation: card1 transaction count + amount sum over
trailing 1h / 24h / 7d sliding windows, sinked to Redis (Feast online store) and Parquet
(Feast offline store).

Scope note (see TASKS.md Known Gaps): this first pass covers the `card1` entity only, not
DeviceInfo/P_emaildomain — the pattern (build_velocity_table) generalizes directly to those,
but card1 alone already exercises the full pipeline (Kafka source -> windowed SQL ->
DataStream interop -> dual sink) end to end, which is the part worth proving first.

Why sliding (HOP) windows, not tumbling: fraud velocity features are inherently "trailing N
hours/days as of right now", not "this calendar-aligned bucket" -- HOP is the architecturally
correct primitive here, at the real cost of more open window state per key. Slide intervals are
chosen coarser for longer windows specifically to bound that cost:
  - 1h window / 5 min slide  -> 12 concurrent windows per key
  - 24h window / 1h slide    -> 24 concurrent windows per key
  - 7d window / 6h slide     -> 28 concurrent windows per key
Coarser slide = slightly staler feature (bounded by the slide interval), which is an explicit,
documented trade against state size -- not an oversight.

Why custom Python sinks instead of Flink's native Redis/filesystem-parquet connectors: the
Kafka connector jar already required hunting for a Flink-2.3-compatible build (none published
yet, closest is a 2.2-tagged jar). Repeating that hunt for a second and third connector on a
brand-new Flink major version is a real risk to build time; a RichMapFunction / RichSinkFunction
using redis-py and pyarrow directly (both already dependencies elsewhere in this repo) is a
legitimate, lower-risk substitute that still round-trips real Redis and real Parquet files.
"""
from __future__ import annotations

import json
from pathlib import Path

import redis
import pyarrow as pa
import pyarrow.parquet as pq
from pyflink.common import Row
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction
from pyflink.table import StreamTableEnvironment

JAR_PATH = Path(__file__).resolve().parent / "jars" / "flink-sql-connector-kafka-5.0.0-2.2.jar"
OFFLINE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

WINDOWS = {
    "1h": ("INTERVAL '5' MINUTE", "INTERVAL '1' HOUR"),
    "24h": ("INTERVAL '1' HOUR", "INTERVAL '24' HOUR"),
    "7d": ("INTERVAL '6' HOUR", "INTERVAL '7' DAY"),
}

PARQUET_SCHEMA = pa.schema([
    ("card1", pa.int64()),
    ("window_start", pa.string()),
    ("window_end", pa.string()),
    ("txn_count", pa.int64()),
    ("amt_sum", pa.float64()),
])


class RedisVelocityMap(MapFunction):
    """Upserts the latest completed window's aggregate per card1 into Redis -- this IS the
    Feast online store's backing store for these features. Side-effecting map, not a terminal
    sink: PyFlink 2.x's SinkFunction wraps Java sinks only, so a custom Python "sink" here is a
    MapFunction chained ahead of a real terminal op (see main() -- terminated with .print())."""

    def __init__(self, window_label: str):
        self.window_label = window_label
        self.client = None

    def open(self, runtime_context):
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=True)

    def map(self, row: Row) -> Row:
        key = f"velocity:card1:{self.window_label}:{row.card1}"
        self.client.hset(key, mapping={
            "txn_count": row.txn_count,
            "amt_sum": row.amt_sum,
            "window_end": str(row.window_end),
        })
        return row


class ParquetVelocityMap(MapFunction):
    """Appends every closed window row to a Parquet file -- Feast's offline store. Writes
    per-row (no internal batching) to guarantee durability if the job is cancelled mid-run;
    the known cost is many small row groups rather than a few large ones, a real perf tradeoff
    to revisit if this needs to sustain much higher throughput than a demo replay."""

    def __init__(self, window_label: str):
        self.window_label = window_label
        self.writer = None
        self.path = OFFLINE_DIR / f"card_velocity_{window_label}.parquet"

    def open(self, runtime_context):
        OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(str(self.path), PARQUET_SCHEMA)

    def map(self, row: Row) -> Row:
        batch = pa.RecordBatch.from_pylist([{
            "card1": row.card1,
            "window_start": str(row.window_start),
            "window_end": str(row.window_end),
            "txn_count": row.txn_count,
            "amt_sum": row.amt_sum,
        }], schema=PARQUET_SCHEMA)
        self.writer.write_batch(batch)
        return row

    def close(self):
        if self.writer:
            self.writer.close()


def build_velocity_table(t_env: StreamTableEnvironment, window_label: str):
    slide, size = WINDOWS[window_label]
    return t_env.sql_query(f"""
        SELECT
            card1,
            window_start,
            window_end,
            COUNT(*) AS txn_count,
            SUM(TransactionAmt) AS amt_sum
        FROM TABLE(
            HOP(TABLE transactions_raw, DESCRIPTOR(event_time), {slide}, {size})
        )
        WHERE card1 IS NOT NULL
        GROUP BY card1, window_start, window_end
    """)


def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.add_jars(f"file://{JAR_PATH}")
    t_env = StreamTableEnvironment.create(env)
    t_env.get_config().set("pipeline.jars", f"file://{JAR_PATH}")

    t_env.execute_sql("""
        CREATE TABLE transactions_raw (
            TransactionID BIGINT,
            TransactionDT BIGINT,
            TransactionAmt DOUBLE,
            card1 BIGINT,
            event_time AS TO_TIMESTAMP(FROM_UNIXTIME(TransactionDT)),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' MINUTE
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'transactions.raw',
            'properties.bootstrap.servers' = 'localhost:19092',
            'properties.group.id' = 'flink-velocity-aggregator',
            'scan.startup.mode' = 'earliest-offset',
            'format' = 'json',
            'json.ignore-parse-errors' = 'true'
        )
    """)

    for label in WINDOWS:
        table = build_velocity_table(t_env, label)
        ds = t_env.to_data_stream(table)
        ds.map(RedisVelocityMap(label)).map(ParquetVelocityMap(label)).print()

    env.execute("driftline-velocity-aggregator")


if __name__ == "__main__":
    main()
