# Phase 2 — Streaming Infrastructure: results and findings

Full pipeline verified end-to-end on the GCP VM (`driftline-vm`, e2-standard-4, 16GB/4vCPU):
**replay producer → Redpanda (Kafka API) → PyFlink windowed aggregation (card1, HOP 1h/24h/7d)
→ dual sink (Redis online store + Parquet offline store) → Feast (entities/feature views,
materialize, online reads)**.

## What's real vs. simulated
Real: the Redpanda broker/partitions/consumer-groups, the PyFlink stateful windowed
aggregation, the Feast online (Redis) and offline (Parquet) stores and the skew between them,
every measured latency number below. Simulated: traffic is a **replay** of historical IEEE-CIS
transactions at a controllable rate, not live production traffic — stated here and in the
top-level README.

## Producer/consumer correctness
- `producer/verify_replay.py`: 5,000/5,000 unique TransactionIDs, 0 duplicates, 0 per-partition
  ordering violations across all 6 partitions (partitioned by `card1` — Kafka only guarantees
  order within a partition/key, not globally across the topic; that's the honest property tested,
  not global ordering, which key-partitioning doesn't provide by design).
- Producer sustained **~150-250 events/sec** actual (well under the requested rate at small
  batch sizes) on this hardware — a real measured number, not the configured target.

## Windowed aggregation (PyFlink 2.3.0, HOP windows over card1)
- 1h window / 5 min slide, 24h window / 1h slide, 7d window / 6h slide — coarser slide on
  longer windows trades feature freshness for bounded per-key open-window state.
- Verified against an 8,000-event replay: 12,746 window-close events for the 1h view alone,
  correctly bucketed and monotonically increasing window boundaries.
- **Scope note:** this covers the `card1` entity only, not DeviceInfo/P_emaildomain (see Known
  Gaps in TASKS.md) — the pattern generalizes directly, card1 alone exercises the full pipeline.

## Feast feature store
- Entities/feature views registered for `card_velocity_{1h,24h,7d}`, `feast apply` +
  `feast materialize` verified against the Redis online store and Parquet offline store.
- Online read sanity check (`test_online_read.py`): real card1 keys return real feature values;
  an unknown card1 correctly returns `None`.

## Training-serving skew — two comparisons, one trivial and one real
1. **`skew_test.py`** (Feast materialize vs. its own Parquet source): **0/1500 mismatches**.
   This can only ever pass — `materialize` is a straight copy of the offline data into Redis,
   nothing structural can make it diverge. Included for completeness, not as the real skew story.
2. **`skew_test_realtime.py`** (the two write paths the Flink job itself produces per event — a
   real-time per-event Redis `hset` vs. a 200-row-buffered Parquet flush): **240/2,772 (window,
   card1) pairs mismatched (~8.7%)**, reproduced across two independent runs (108/3,406 on the
   first 5,000-event run, 240/2,772 on the 8,000-event rerun). **Root cause, found and explained,
   not fabricated:** the Parquet sink batches 200 rows before writing a part-file (a deliberate
   durability-vs-latency tradeoff, see below), while the Redis sink writes per-event. A card1's
   most recent event can be visible online before its window-close row has been flushed to
   Parquet — the *reverse* of the usual "online lags offline" assumption, because here online is
   the low-latency path and offline is the batched one.

## Feature freshness lag (event `produced_at` → Redis online-store write, wall-clock)
Measured on an 8,000-event replay at ~150-170 events/sec actual producer rate:

| Window | n samples | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| 1h | 12,746 | 14.7s | 23.9s | 25.6s | 27.1s |
| 24h | 10,637 | 14.1s | 27.9s | 32.7s | 37.0s |
| 7d | 2,741 | 7.4s | 25.7s | 34.9s | 37.2s |

This lag is dominated by the producer's own send rate (events must arrive before their window
can close) plus the watermark's 5-minute allowed lateness, not by Redis write latency itself —
worth being able to explain the breakdown, not just the number, in an interview.

## Real bugs found and fixed during this phase (kept, not silently patched)
1. **Parquet durability bug:** a single incrementally-appended `ParquetWriter` produces an
   **invalid file** (no footer — "Parquet magic bytes not found") if the process is killed via
   SIGTERM rather than shut down gracefully, which is exactly how a streaming job actually gets
   stopped. Fixed by writing small self-contained part-files (buffer 200 rows, flush as a
   complete `pq.write_table` call) — bounds data loss to one unflushed buffer instead of losing
   the entire file.
2. **Silently dropped column:** the producer embedded `produced_at` in every event for the
   freshness-lag metric, but the Flink source table's `CREATE TABLE` never declared that column,
   so it was silently discarded by the JSON format parser. Found while trying to implement the
   metric the source doc calls for; fixed by declaring it and carrying `MAX(produced_at)` through
   the windowed aggregation.
3. **Orphaned JVM child process:** PyFlink's Python driver spawns a Java process (the actual
   Flink minicluster) as a child; killing the driver via `timeout`/SIGTERM does not reliably kill
   that child. An earlier smoke-test invocation sat running for roughly an hour undetected,
   consuming CPU, until found via `ps aux` and killed manually. Worth knowing for anyone running
   PyFlink locally: check for orphaned `java ... PythonGatewayServer` processes, don't assume
   `timeout` cleaned everything up.
4. **Small-batch HOP window emission threshold:** a batch of only 3,000 events (especially two
   *duplicate* 3,000-event batches, since the producer's `--limit` always takes the same first N
   rows rather than continuing from a prior run) doesn't span enough event-time for even the
   first sliding window to close, and produces zero output with no error — easy to mistake for a
   pipeline bug rather than a data-volume/event-time-span issue. Confirmed by scaling to 8,000
   events, which reliably produced output.
5. **Stale consumer-group offsets across topic delete/recreate:** Redpanda's topic delete+create
   cycle didn't reliably reset in time for an immediately-following `rpk topic create`
   (`TOPIC_ALREADY_EXISTS` on the first retry), and a reused Flink consumer group id risks
   resuming from a stale offset rather than honoring `scan.startup.mode=earliest-offset`. Worked
   around with a configurable `FLINK_GROUP_ID` env var for isolated test runs.

## What's NOT done yet (see TASKS.md Known Gaps)
- DeviceInfo/P_emaildomain entity velocity features (card1 only, this phase).
- Spark Structured Streaming alternative path (PyFlink only, as permitted — the source doc says
  "claim both truthfully" and this documents the omission rather than falsely claiming both).
- Feast contract test (schema-change-fails-CI) — Phase 6 item, not yet built.
- Consumer-lag dashboarding under sustained load (verified `rpk group` mechanics conceptually,
  not yet load-tested at Phase 4's throughput scale).
