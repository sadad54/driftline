"""Overload behaviour of the scorer: probes stay answerable and excess load is shed.

These tests run the real FastAPI app on a real uvicorn server (in a background thread) with the
ONNX session replaced by a fake whose `run` blocks until the test releases it. That makes
"every scorer thread is busy" a state the test can hold still and inspect, with no timing
guesswork and no model file.

The regression they guard: in the Phase 6 load test a saturated pod could not answer its health
probe and kubelet restarted it (results/phase6_production.md).
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import numpy as np
import pytest
import uvicorn

from serving import app as scorer

FEATURE_COLUMNS = ["TransactionAmt", "ProductCD"]


class BlockingSession:
    """Stands in for the ONNX session. `run` waits for `release` before returning a score."""

    def __init__(self):
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def run(self, _outputs, _inputs):
        with self._lock:
            self.calls += 1
        assert self.release.wait(timeout=10), "test never released the fake model"
        return [np.array([0]), [{0: 0.9, 1: 0.1}]]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(monkeypatch):
    """The scorer on a live uvicorn server, model faked, admission limit set to 4."""
    session = BlockingSession()
    monkeypatch.setattr(scorer, "_session", session)
    monkeypatch.setattr(
        scorer, "_preprocessing", {"feature_columns": FEATURE_COLUMNS, "category_maps": {"ProductCD": ["W", "C"]}}
    )
    monkeypatch.setattr(scorer, "_feast_store", None)
    monkeypatch.setattr(scorer.limiter, "limit", 4)
    monkeypatch.setattr(scorer.limiter, "inflight", 0)

    port = _free_port()
    # lifespan="off": the startup hook loads the real model from disk, which these tests replace.
    config = uvicorn.Config(scorer.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(base + "/healthz", timeout=0.5)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        raise RuntimeError("test server did not start")

    yield base, session

    session.release.set()
    srv.should_exit = True
    thread.join(timeout=10)


PAYLOAD = {"features": {"TransactionAmt": 42.0, "ProductCD": "W"}}


def _saturate(base: str, session: BlockingSession, n: int) -> list[threading.Thread]:
    """Start n /score calls that will sit inside the (blocked) scorer until released."""
    results: list[int] = []

    def call():
        results.append(httpx.post(base + "/score", json=PAYLOAD, timeout=15).status_code)

    threads = [threading.Thread(target=call) for _ in range(n)]
    for t in threads:
        t.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        if httpx.get(base + "/readyz", timeout=1).json()["inflight"] >= n:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("scorer never reached the expected in-flight count")
    for t in threads:
        t.results = results  # type: ignore[attr-defined]
    return threads


def test_liveness_answers_while_every_scorer_thread_is_blocked(server):
    base, session = server
    threads = _saturate(base, session, 4)

    # The scorer is now completely stuck: all admitted requests are waiting on the model.
    # A liveness probe must still come back well inside kubelet's timeout.
    for _ in range(5):
        t0 = time.perf_counter()
        resp = httpx.get(base + "/healthz", timeout=1.0)
        assert resp.status_code == 200
        assert time.perf_counter() - t0 < 0.5

    session.release.set()
    for t in threads:
        t.join(timeout=10)
    assert threads[0].results == [200, 200, 200, 200]


def test_requests_beyond_the_limit_are_shed_with_503_and_retry_after(server):
    base, session = server
    threads = _saturate(base, session, 4)
    calls_before = session.calls

    t0 = time.perf_counter()
    resp = httpx.post(base + "/score", json=PAYLOAD, timeout=2.0)
    elapsed = time.perf_counter() - t0

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "1"
    assert elapsed < 0.5, "a shed request must be refused immediately, not queued"
    assert session.calls == calls_before, "a shed request must never reach the model"

    session.release.set()
    for t in threads:
        t.join(timeout=10)


def test_readiness_stays_up_when_saturated_and_reports_it(server):
    base, session = server
    threads = _saturate(base, session, 4)

    resp = httpx.get(base + "/readyz", timeout=1.0)
    assert resp.status_code == 200, "being busy is not a reason to leave the Service"
    assert resp.json() == {"ready": True, "saturated": True, "inflight": 4, "limit": 4}

    session.release.set()
    for t in threads:
        t.join(timeout=10)


def test_capacity_is_returned_after_a_burst(server):
    base, session = server
    threads = _saturate(base, session, 4)
    session.release.set()
    for t in threads:
        t.join(timeout=10)

    assert httpx.get(base + "/readyz", timeout=1.0).json()["inflight"] == 0
    assert httpx.post(base + "/score", json=PAYLOAD, timeout=5.0).status_code == 200


def test_slot_is_released_when_scoring_raises(server, monkeypatch):
    base, _ = server

    class Exploding:
        def run(self, *_):
            raise RuntimeError("model blew up")

    monkeypatch.setattr(scorer, "_session", Exploding())
    for _ in range(6):  # more than the limit of 4: leaked slots would turn these into 503s
        assert httpx.post(base + "/score", json=PAYLOAD, timeout=5.0).status_code == 500
    assert httpx.get(base + "/readyz", timeout=1.0).json()["inflight"] == 0


def test_readiness_fails_until_the_model_is_loaded(server, monkeypatch):
    base, _ = server
    monkeypatch.setattr(scorer, "_session", None)

    assert httpx.get(base + "/readyz", timeout=1.0).status_code == 503
    assert httpx.get(base + "/healthz", timeout=1.0).status_code == 200, "not ready is not the same as dead"
    assert httpx.post(base + "/score", json=PAYLOAD, timeout=5.0).status_code == 503


def test_shed_requests_are_counted_in_metrics(server):
    base, session = server
    threads = _saturate(base, session, 4)
    for _ in range(3):
        httpx.post(base + "/score", json=PAYLOAD, timeout=2.0)

    metrics = httpx.get(base + "/metrics", timeout=2.0).text
    shed_line = next(line for line in metrics.splitlines() if line.startswith("driftline_score_shed_total"))
    assert float(shed_line.split()[-1]) >= 3
    assert "driftline_score_inflight 4.0" in metrics

    session.release.set()
    for t in threads:
        t.join(timeout=10)
