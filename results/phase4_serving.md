# Phase 4 — Serving: results and findings

## ONNX export
Attempted the Phase 1 categorical-native XGBoost model (`enable_categorical=True`) first — it
**genuinely fails** to convert via `onnxmltools`: `RecursionError: maximum recursion depth
exceeded`. This confirms the risk flagged before attempting it (ONNX's tree-ensemble op expects
plain numeric input, not pandas category-dtype handling), rather than assuming either outcome.

**Fallback: a numeric "serving" variant** — same 434 features, categoricals ordinal-encoded
(`.cat.codes`) instead of native category dtype, fit on a plain numpy array (fitting on a
DataFrame makes XGBoost store real column names like `'V258'` as `feature_names`, which
onnxmltools' converter can't parse — a second real bug found and fixed along the way, not
assumed).

- **ONNX-vs-native parity: max abs diff 2.98e-7** (float32 precision noise) — PASS.
- **Interesting side finding, not the headline result:** the ordinal-encoded serving variant
  scores PR-AUC 0.5206 vs. the categorical-native model's 0.4776 — notably *higher*. Reported as
  observed, not root-caused further here (plausible causes: different effective split
  granularity from ordinal vs. native categorical handling); worth investigating if pursuing
  this project further, logged rather than either ignored or overclaimed as a deliberate
  improvement.

## FastAPI scoring service
`serving/app.py`: loads the ONNX model + persisted category-encoding metadata at startup,
`/score` accepts a raw transaction, optionally looks up Feast online velocity features by
`card1` (proving connectivity — Feast fetch succeeded: `velocity_features_available: true` on a
real test call), runs ONNX Runtime inference, returns a fraud score.

**Named honestly:** velocity features are fetched but not yet part of the model's input schema —
the Phase 1 XGBoost was trained on raw IEEE-CIS columns only, not the Flink-computed velocity
aggregates. The response field is `velocity_features_available`, not `_used`, so the API doesn't
imply an effect that isn't there yet. Wiring them into a retrained model is a real next step,
logged in Known Gaps.

Single-request sanity check (real test-split transaction): `fraud_score=0.107`,
`latency_ms=20.98` (cold).

## Load test (Locust, 50 concurrent users, 60s, against the single-process uvicorn dev server)
| Metric | Value |
|---|---|
| Total requests | 10,018 |
| Failures | 0 |
| Sustained throughput | ~170-200 req/s |
| p50 latency | 240 ms |
| p95 latency | 380 ms |
| p99 latency | 490 ms |
| max latency | 640 ms |

**Real, explainable finding, not just a number:** single-request latency was ~21ms, but p50 under
50-concurrent-user load is 240ms — an order of magnitude higher. This is queueing, not slower
inference: a single uvicorn worker process serializes CPU-bound ONNX inference work through one
event loop, so it doesn't parallelize across this VM's 4 vCPUs under concurrent load. The direct
lever is more uvicorn workers (`--workers N`) or more replicas behind a load balancer (the k8s
deployment in `k8s/scorer-deployment.yaml` runs 2 replicas with an HPA up to 6, for exactly this
reason) — not implemented/re-benchmarked with multiple workers here for time, logged as the
concrete next step rather than "needs optimization."

## Analyst-budget precision@k simulation
Per-day precision@k across the 42-day test period (mean daily volume ~2,812 transactions),
using the persisted Phase 1 XGBoost model:

| Analyst budget | Reviews/day (mean) | Mean precision@k | Min | Max |
|---|---|---|---|---|
| 0.5% of daily volume | ~14 | **87.7%** | 26.7% | 100% |
| 1.0% | ~28 | **75.6%** | 33.3% | 100% |
| 2.0% | ~56 | **59.6%** | 32.8% | 100% |
| 5.0% | ~141 | **37.1%** | 16.9% | 58.1% |

This is the real, quantified answer to "10,000 flags/day, 200 analysts (2% budget)": at a 2%
review budget on this data's volume, an analyst team should expect ~60% of what they review to
actually be fraud, with real day-to-day variance (33%-100%) worth being able to discuss, not
just the mean.

## k3d deployment
Applied `k8s/scorer-deployment.yaml` to a real k3d cluster on this VM (2 replicas, HPA 2-6 on
70% CPU, PodDisruptionBudget minAvailable=1, liveness/readiness probes). Both pods came up
`Running`/`1/1 Ready`; verified via `kubectl port-forward` + a real scoring request — **the
fraud_score matched the docker-compose deployment exactly** (0.10728633403778076), confirming
correct, deterministic inference inside the cluster.

**Two real issues found, not hidden:**
1. **`velocity_features_available: false`** in the k3d pod (vs. `true` in docker-compose). The
   k3d cluster's pod network is a separate Docker network (`k3d-driftline`) from
   docker-compose's default network (`driftline_default`) — `feature_store.yaml`'s
   `localhost:6379` inside a k3d pod resolves to the pod itself, not the docker-compose Redis
   container. This is a genuine, expected consequence of the two deployment paths living on
   different networks, not wired together here. Fix (not applied, logged in Known Gaps): either
   deploy Redis inside the k3d cluster too, or give the scorer pods `hostNetwork: true` the same
   way the docker-compose scorer service uses `network_mode: host`.
2. **4.78-second latency on that first k3d request** (vs. ~21ms for the equivalent
   docker-compose/direct request) — consistent with the Feast/Redis connection attempt hanging
   on a TCP timeout rather than failing fast, given issue #1 above. Not root-caused further
   (would need to trace the actual redis-py connect timeout behavior inside the pod); logged as
   the likely cause rather than asserted as certain.

## What's not done yet (Known Gaps)
- Multi-worker/replica latency re-benchmark (the direct fix for the queueing finding above).
- k3d/docker-compose network bridging for Feast/Redis (see above) — the scorer works correctly
  in both deployment paths, but Feast connectivity only works in the docker-compose one right now.
- End-to-end smoke test (producer → Redpanda → Flink → Feast → scorer → `transactions.scored`)
  not yet run as one connected pipeline — each stage verified independently so far.
- Velocity features fetched but not wired into the scoring model's input (see above).
