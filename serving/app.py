"""FastAPI fraud-scoring service.

Loads the ONNX XGBoost serving model (scripts/export_onnx.py's output) plus its preprocessing
metadata, exposes a /score endpoint that accepts a raw transaction record, optionally enriches it
with Feast online velocity features (card_velocity_1h/24h/7d -- Phase 2), and returns a fraud
score via ONNX Runtime.

Why ONNX Runtime rather than loading the XGBoost model directly: this is genuinely the serving
path -- the point of the ONNX export -- and it's also meaningfully faster for single-row
inference than the full XGBoost Python API, which matters for the p99 latency number Phase 4
measures.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"

# --- Overload behaviour ------------------------------------------------------------------------
# The Phase 6 load test found that a saturated pod stopped answering its own health probe and was
# restarted by kubelet while it was healthy (results/phase6_production.md). Three changes fix the
# cause, and they are deliberately in the application, not only in the probe settings:
#
#   1. Scoring runs on its own small thread pool. The event loop only parses requests and answers
#      probes, so a probe never waits behind scoring work.
#   2. Admission control. At most DRIFTLINE_MAX_INFLIGHT /score requests are admitted at once.
#      Anything beyond that is refused immediately with 503 + Retry-After, before its body is
#      read, so the queue (and therefore latency) is bounded and the pod sheds load it cannot serve.
#   3. Liveness and readiness are separate endpoints. /healthz only says "the process is
#      responsive". /readyz says "this pod can serve" (model loaded). Being busy is never a
#      reason to fail liveness.
SCORER_THREADS = int(os.getenv("DRIFTLINE_SCORER_THREADS", "2"))
MAX_INFLIGHT = int(os.getenv("DRIFTLINE_MAX_INFLIGHT", "16"))
ORT_THREADS = int(os.getenv("DRIFTLINE_ORT_THREADS", "1"))
RETRY_AFTER_SECONDS = "1"


class AdmissionLimiter:
    """Counts admitted /score requests. Only ever touched from the event loop thread, so a plain
    integer is enough -- no lock."""

    def __init__(self, limit: int):
        self.limit = limit
        self.inflight = 0

    def try_acquire(self) -> bool:
        if self.inflight >= self.limit:
            return False
        self.inflight += 1
        return True

    def release(self) -> None:
        self.inflight -= 1

    @property
    def saturated(self) -> bool:
        return self.inflight >= self.limit


limiter = AdmissionLimiter(MAX_INFLIGHT)
_executor = ThreadPoolExecutor(max_workers=SCORER_THREADS, thread_name_prefix="scorer")

app = FastAPI(title="Driftline Fraud Scoring Service")

SCORE_REQUESTS = Counter("driftline_score_requests_total", "Total /score requests")
SCORE_LATENCY = Histogram(
    "driftline_score_latency_seconds", "End-to-end /score handler latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
VELOCITY_HIT = Counter("driftline_velocity_feature_hits_total", "Requests where Feast velocity features were available")
FRAUD_SCORE = Histogram(
    "driftline_fraud_score", "Distribution of returned fraud scores",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)

SCORE_SHED = Counter(
    "driftline_score_shed_total", "/score requests refused with 503 because the pod was at capacity"
)
SCORE_INFLIGHT = Gauge("driftline_score_inflight", "/score requests currently admitted (running or queued)")

_SHED_BODY = b'{"detail":"scorer at capacity, retry shortly"}'


class AdmissionControlMiddleware:
    """Refuses /score when the pod is at capacity, before the request body is read or parsed.

    Written as a raw ASGI middleware on purpose: rejecting inside the route handler would still
    make the event loop read and validate a full feature payload for a request it is about to
    refuse, which is exactly the work an overloaded pod cannot afford."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/score" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        if not limiter.try_acquire():
            SCORE_SHED.inc()
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", RETRY_AFTER_SECONDS.encode()),
                    (b"content-length", str(len(_SHED_BODY)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": _SHED_BODY})
            return
        SCORE_INFLIGHT.set(limiter.inflight)
        try:
            await self.app(scope, receive, send)
        finally:
            limiter.release()
            SCORE_INFLIGHT.set(limiter.inflight)


app.add_middleware(AdmissionControlMiddleware)

_session: Optional[ort.InferenceSession] = None
_preprocessing: Optional[dict] = None
_feast_store = None


@app.on_event("startup")
def load_model():
    global _session, _preprocessing, _feast_store
    onnx_path = MODEL_DIR / "xgboost_serving.onnx"
    prep_path = MODEL_DIR / "serving_preprocessing.json"
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX model not found at {onnx_path} -- run scripts/export_onnx.py first")
    # One intra-op thread: single-row tree inference gains nothing from more, and ONNX Runtime's
    # default (one spinning thread per host core) fights the scorer threads for a pod's CPU limit.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    _session = ort.InferenceSession(str(onnx_path), sess_options=opts)
    with open(prep_path) as f:
        _preprocessing = json.load(f)

    try:
        from feast import FeatureStore
        _feast_store = FeatureStore(repo_path=str(Path(__file__).resolve().parents[1] / "feature_store"))
    except Exception as e:
        print(f"Feast store unavailable ({e}); velocity features will be zero-filled")
        _feast_store = None


class Transaction(BaseModel):
    features: dict  # raw feature dict, keys matching preprocessing["feature_columns"]
    card1: Optional[int] = None


class ScoreResponse(BaseModel):
    fraud_score: float
    latency_ms: float
    velocity_features_available: bool


def _encode_row(features: dict) -> np.ndarray:
    cols = _preprocessing["feature_columns"]
    category_maps = _preprocessing["category_maps"]
    row = np.zeros(len(cols), dtype=np.float32)
    for i, col in enumerate(cols):
        val = features.get(col)
        if col in category_maps:
            cats = category_maps[col]
            row[i] = float(cats.index(val)) if val in cats else -1.0
        elif val is None:
            row[i] = -1.0
        else:
            row[i] = float(val)
    return row.reshape(1, -1)


@app.post("/score", response_model=ScoreResponse)
async def score(txn: Transaction):
    if _session is None:
        raise HTTPException(503, "model not loaded")
    # t0 is taken before the hand-off, so the latency metric includes time spent queued for a
    # scorer thread -- the number a caller actually experiences.
    t0 = time.perf_counter()
    return await asyncio.get_running_loop().run_in_executor(_executor, _score_sync, txn, t0)


def _score_sync(txn: Transaction, t0: float) -> ScoreResponse:
    features = dict(txn.features)
    velocity_used = False

    if txn.card1 is not None and _feast_store is not None:
        try:
            result = _feast_store.get_online_features(
                features=[
                    "card_velocity_1h:txn_count", "card_velocity_1h:amt_sum",
                    "card_velocity_24h:txn_count", "card_velocity_7d:txn_count",
                ],
                entity_rows=[{"card1": txn.card1}],
                full_feature_names=True,
            ).to_dict()
            # Not part of the current XGBoost feature schema (Phase 1 trained without them) --
            # fetched and available for a future retrain, not silently dropped. Logged, not used
            # in scoring yet: see Known Gaps.
            velocity_used = result["card_velocity_1h__txn_count"][0] is not None
        except Exception:
            pass

    x = _encode_row(features)
    probs = _session.run(None, {"input": x})[1]
    fraud_score = float(probs[0][1])

    elapsed = time.perf_counter() - t0
    SCORE_REQUESTS.inc()
    SCORE_LATENCY.observe(elapsed)
    FRAUD_SCORE.observe(fraud_score)
    if velocity_used:
        VELOCITY_HIT.inc()

    return ScoreResponse(fraud_score=fraud_score, latency_ms=elapsed * 1000, velocity_features_available=velocity_used)


@app.get("/healthz")
async def healthz():
    """Liveness: is the process responsive? Deliberately checks nothing else. It is async and does
    no work, so it is answered by the event loop even when every scorer thread is busy."""
    return {"status": "alive"}


@app.get("/readyz")
async def readyz(response: Response):
    """Readiness: can this pod serve traffic? False only while the model is not loaded.

    Saturation is reported in the body but does not fail readiness. An overloaded pod already
    refuses excess work with 503 + Retry-After; also pulling it out of the Service would push its
    share of traffic onto the other pods and tip them over in turn."""
    ready = _session is not None
    if not ready:
        response.status_code = 503
    return {"ready": ready, "saturated": limiter.saturated, "inflight": limiter.inflight, "limit": limiter.limit}


@app.get("/health")
async def health():
    """Kept for existing scripts and dashboards. Same cost as /healthz."""
    return {"status": "ok", "model_loaded": _session is not None}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
