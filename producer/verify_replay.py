"""Verify replay producer correctness: consume everything from `transactions.raw` and check
(1) no dropped/duplicated events, (2) per-partition ordering is preserved.

Note on ordering: events are keyed by `card1` so all events for the same card always land in the
same partition (needed later for Flink's keyed per-entity windows). Kafka only guarantees order
*within* a partition, not globally across the topic's 6 partitions — so the correct property to
check is "each partition's consumed order is non-decreasing in TransactionDT", not "the whole
topic reproduces global send order". Testing for the wrong (global) guarantee would either fail
on a correct system or hide a real bug; testing for the right one is itself a substantive
interview point about how Kafka ordering actually works.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from kafka import KafkaConsumer, TopicPartition

TOPIC = "transactions.raw"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:19092")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    consumer = KafkaConsumer(
        bootstrap_servers=args.bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="verify-replay",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=args.timeout_ms,
    )
    partitions = [TopicPartition(TOPIC, p) for p in consumer.partitions_for_topic(TOPIC)]
    consumer.assign(partitions)
    consumer.seek_to_beginning(*partitions)

    seen_txn_ids = set()
    duplicates = 0
    per_partition_dt = defaultdict(list)

    for msg in consumer:
        txn_id = msg.value["TransactionID"]
        if txn_id in seen_txn_ids:
            duplicates += 1
        seen_txn_ids.add(txn_id)
        per_partition_dt[msg.partition].append(msg.value["TransactionDT"])

    consumer.close()

    total = len(seen_txn_ids) + duplicates
    print(f"Consumed {total} messages ({len(seen_txn_ids)} unique TransactionID, {duplicates} duplicates)")
    print(f"Expected: {args.expected_count}")

    order_violations = 0
    for partition, dts in per_partition_dt.items():
        violations = sum(1 for a, b in zip(dts, dts[1:]) if b < a)
        order_violations += violations
        print(f"  partition {partition}: {len(dts)} events, {violations} order violations")

    ok = (
        len(seen_txn_ids) == args.expected_count
        and duplicates == 0
        and order_violations == 0
    )
    print("\n" + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
