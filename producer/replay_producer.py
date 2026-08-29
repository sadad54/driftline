"""Replays IEEE-CIS transactions onto the `transactions.raw` Kafka topic in strict
TransactionDT order, at a configurable rate.

This is explicitly a REPLAY of historical data, not live traffic — the source doc's own honesty
principle: real broker, real partitions/consumer-groups/offsets, but the traffic itself is
historical data played back at a controllable speed, and that's stated plainly here and in the
README rather than dressed up as something it isn't.

Usage:
    python producer/replay_producer.py --events-per-sec 200
    python producer/replay_producer.py --events-per-sec 500 --limit 50000  # smoke test
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from kafka import KafkaProducer

from driftline.data import load_ieee_cis

TOPIC = "transactions.raw"

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    print("\nCaught interrupt, finishing current batch and stopping...", file=sys.stderr)
    _stop = True


def row_to_message(row: dict) -> dict:
    """JSON-safe payload: NaN -> None, everything else passed through.

    Keeps the full feature row (not just entity-graph keys) because the Phase 4 scoring service
    needs the complete feature vector at scoring time — Feast supplies the *derived* velocity
    features, but the base transaction fields have to travel with the event itself.
    """
    return {k: (None if v != v else v) for k, v in row.items()}  # v != v is the NaN check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:19092")
    parser.add_argument("--events-per-sec", type=float, default=200.0)
    parser.add_argument("--limit", type=int, default=None, help="cap total events (for smoke tests)")
    parser.add_argument("--key-field", default="card1", help="partition key field")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    print("Loading IEEE-CIS (sorted by TransactionDT)...")
    df = load_ieee_cis()
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"  {len(df):,} events to replay at {args.events_per_sec}/sec "
          f"({len(df) / args.events_per_sec:.1f}s wall-clock)")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        linger_ms=5,
        acks="all",
    )

    interval = 1.0 / args.events_per_sec
    sent, last_report = 0, time.time()
    t_start = time.time()

    records = df.to_dict(orient="records")
    for row in records:
        if _stop:
            break
        payload = row_to_message(row)
        # produced_at: wall-clock ingestion time, distinct from TransactionDT (the historical
        # event time) — Flink windows on TransactionDT, not this, but this lets us measure
        # feature-freshness lag (produced_at -> Feast online-store write time) honestly.
        payload["produced_at"] = time.time()
        key = row.get(args.key_field)
        producer.send(TOPIC, key=key, value=payload)
        sent += 1

        if time.time() - last_report > 5:
            elapsed = time.time() - t_start
            print(f"  sent {sent:,}/{len(df):,} ({sent / elapsed:.1f} actual events/sec)")
            last_report = time.time()

        time.sleep(interval)

    producer.flush()
    producer.close()
    elapsed = time.time() - t_start
    print(f"Done. Sent {sent:,} events in {elapsed:.1f}s ({sent / elapsed:.1f} events/sec actual).")


if __name__ == "__main__":
    main()
