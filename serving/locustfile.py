"""Locust load test against the /score endpoint. Real transaction rows (sampled from the test
split) are used as payloads, not synthetic/empty requests -- so the load test also exercises the
actual categorical-encoding path, not just an empty round trip.

Usage (from the VM, scorer running on localhost:8000):
    locust -f serving/locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 10 --run-time 60s --headless \
        --csv results/locust_run
"""
import json
import random
from pathlib import Path

from locust import HttpUser, task, between

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_requests.json"

with open(SAMPLE_PATH) as f:
    SAMPLE_REQUESTS = json.load(f)


class ScorerUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def score_transaction(self):
        payload = random.choice(SAMPLE_REQUESTS)
        self.client.post("/score", json=payload)
