"""End-to-end smoke test: consume real replayed events from `transactions.raw`, score each via
the live FastAPI scoring service (real HTTP call, not an in-process import), publish
{TransactionID, fraud_score} to `transactions.scored`, then read them back to confirm the whole
loop actually closes.

Requires: the replay producer to have already put events on transactions.raw (e.g.
`python producer/replay_producer.py --limit 200`), and the scorer running at --scorer-url.
"""
import argparse
import json
import time

import requests
from kafka import KafkaConsumer, KafkaProducer

RAW_TOPIC = "transactions.raw"
SCORED_TOPIC = "transactions.scored"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:19092")
    parser.add_argument("--scorer-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    args = parser.parse_args()

    consumer = KafkaConsumer(
        RAW_TOPIC,
        bootstrap_servers=args.bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="e2e-smoke-test",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=args.timeout_ms,
    )
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    scored = 0
    latencies = []
    for msg in consumer:
        txn = msg.value
        card1 = txn.get("card1")
        payload = {"features": txn, "card1": card1}

        t0 = time.time()
        resp = requests.post(f"{args.scorer_url}/score", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        latencies.append(time.time() - t0)

        producer.send(SCORED_TOPIC, value={
            "TransactionID": txn["TransactionID"],
            "fraud_score": result["fraud_score"],
            "scored_at": time.time(),
        })
        scored += 1
        if scored >= args.limit:
            break

    producer.flush()
    consumer.close()
    print(f"Scored and published {scored} events. "
          f"HTTP round-trip latency: mean={sum(latencies)/len(latencies)*1000:.1f}ms" if latencies else "No events found")

    # read back from transactions.scored to prove the loop actually closed
    verify_consumer = KafkaConsumer(
        SCORED_TOPIC,
        bootstrap_servers=args.bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="e2e-smoke-test-verify",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=args.timeout_ms,
    )
    verified = list(verify_consumer)
    verify_consumer.close()
    print(f"Read back {len(verified)} messages from {SCORED_TOPIC}")
    if verified:
        print(f"  sample: {verified[0].value}")

    assert scored > 0, "no events were scored -- is transactions.raw empty?"
    assert len(verified) >= scored, "fewer scored messages readable than were published"
    print("\nPASS: producer -> Redpanda -> scorer -> transactions.scored loop verified end-to-end")


if __name__ == "__main__":
    main()
