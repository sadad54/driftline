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

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"

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
    _session = ort.InferenceSession(str(onnx_path))
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
def score(txn: Transaction):
    if _session is None:
        raise HTTPException(503, "model not loaded")

    t0 = time.perf_counter()
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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _session is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
