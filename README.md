# Driftline

**A streaming fraud-detection platform: real-time-shaped ingestion, a graph-augmented model, and
drift-triggered retraining that actually recovers a real performance collapse — built and
measured end-to-end, with every honest failure kept in the record alongside the wins.**

## What's real, and what's a replay — stated plainly, up front

**Real:** the Redpanda broker, its partitions/consumer-groups/offsets; the PyFlink windowed
stateful aggregation; the Feast online (Redis) and offline (Parquet) stores, and a measured skew
between them; the PyTorch Geometric GraphSAGE entity-graph model with a tested leakage boundary;
the PSI/KS drift detection; the drift-triggered retrain → causal shadow-score → enforced
promotion gate pipeline (every run logged to MLflow); the FastAPI+ONNX scoring service and its
measured latency/throughput; the Kubernetes deployment and its observed HPA scale event.

**Simulated:** "real-time" here means a **replay** of six months of historical IEEE-CIS
transactions through a real broker, at a controllable rate — not live production traffic. Every
number in this repo is honest about that distinction; it's a deliberate differentiator, not
something to hide.

## The headline result

Trained on the earliest month of data, evaluated forward with no retraining, PR-AUC decays from
**0.4761** (first held-out month) to **0.3680** (month 6) — a real 22.7% relative decline, driven
by real drift in the data, not injected noise. A weekly PSI-triggered retraining loop, walked
forward across the same six months with a **causal shadow-scored promotion gate** (every
candidate model evaluated on a held-out tail of its own training window before being trusted —
never on the future), recovers a mid-stream collapse from PR-AUC **0.2635 to 0.5211** — a
**+97.8% relative recovery in a single week** — and both retrains it triggered were promoted.

*(Full numbers, every one sourced from a committed `results/*.json` file:
[`metrics/README.md`](metrics/README.md).)*

## Architecture

```
IEEE-CIS transactions (590,540 rows, 6 months, real Vesta data)
        │
  replay_producer.py — emits events in TransactionDT order at a controllable rate
        ▼
   Redpanda (Kafka API) — topic: transactions.raw, 6 partitions, keyed by card1
        ▼
  PyFlink — HOP (sliding) windowed velocity aggregation: 1h / 24h / 7d
   ├─ online sink  → Redis (Feast online store + a hand-rolled real-time keyspace)
   └─ offline sink → Parquet part-files (Feast offline store)
        ▼
  Scoring service (FastAPI + ONNX Runtime)
   ├─ XGBoost (ONNX-exported; the categorical-native model doesn't convert -- documented)
   └─ [GraphSAGE ensemble built and measured, not yet wired into serving -- see Known Gaps]
        ▼
   topic: transactions.scored
        ▼
  Monitoring: weekly PSI/KS drift (369 features) + Prometheus/Grafana (scorer metrics)
        ▼
  Drift-triggered retrain: PSI breach → retrain → causal shadow-score → promotion gate → MLflow
        ▼
  Kubernetes (k3d): scorer Deployment, 2-6 replica HPA, PodDisruptionBudget, resource limits
```

Entity graph (Phase 3): 590,540 transaction nodes + 14,811 identity-value nodes
(card1/card2/card3/card5/addr1/addr2/email domains), connected by "has-this-value" edges —
605K nodes, ~8M edges, trained with a **tested** guarantee that the training graph structurally
cannot reference a test-set row.

## What actually happened — the honest parts, not just the wins

This project's whole premise is that every claim is real and reproducible, which means the
ablations that *didn't* work are as much a deliverable as the ones that did:

- **The GraphSAGE + XGBoost ensemble does not beat XGBoost alone** (PR-AUC 0.4687 vs. 0.4751,
  naive rank-average). Root-caused, not shrugged off: equal-weight ensembling punishes a weaker
  model instead of learning how much to trust it, and 5 epochs of CPU-only training genuinely
  undertrains a GNN relative to a fully-tuned 400-tree XGBoost baseline. Same failure mode showed
  up independently with IsolationForest in Phase 1 — a real, repeated pattern, not a one-off.
- **A real "your metrics lied to you" number, not an assertion:** the identical XGBoost config
  scores PR-AUC 0.6974 on a random 80/20 split vs. the honest 0.4751 on a time-ordered one — a
  +0.2222 absolute inflation from nothing but the split strategy.
- **A real training-serving skew, found and explained:** the Flink job's real-time Redis write and
  its 200-row-buffered Parquet write disagree on 8.66% of sampled (card1, window) pairs — online
  runs briefly *ahead* of offline, the reverse of the usual assumption, because of a deliberate
  durability-vs-latency tradeoff in the Parquet sink.
- **A real production reliability finding from actually load-testing Kubernetes:** driving
  100 concurrent users at the k3d-deployed scorer produced a genuine HPA scale event (2→4
  replicas) *and* near-simultaneous liveness-probe failures across pods (single uvicorn worker
  too saturated to answer a trivial health check), causing a real kubelet-triggered restart.
- **The categorical-native XGBoost model genuinely fails ONNX export** (`RecursionError` /
  a feature-name-pattern error in `onnxmltools`, confirmed by trying, not assumed). The serving
  model is a numeric ordinal-encoded variant, documented as a deliberate difference from the
  research model, with parity verified to 2.98e-7.
- **Several real infrastructure bugs found by actually running things**, not written and assumed
  correct: a Parquet-durability bug under SIGTERM, a silently-dropped `produced_at` column, an
  orphaned JVM process that outlived `timeout` by about an hour, a graph node-id collision, CI
  disk exhaustion from an unscoped CUDA-enabled torch install, and a silent numpy upgrade that
  broke XGBoost's internal `np.NaN` usage. Each one is documented at the point it was found in
  `results/phase*.md`, not quietly fixed and forgotten.

## Metrics

Full consolidated numbers, every one with provenance back to a committed script and result file:
**[`metrics/README.md`](metrics/README.md)** (rendered table) /
**[`metrics/summary.json`](metrics/summary.json)** (machine-readable).

Phase-by-phase narrative writeups with the reasoning behind every design decision:
[`results/phase2_streaming.md`](results/phase2_streaming.md) ·
[`results/phase3_graph.md`](results/phase3_graph.md) ·
[`results/phase4_serving.md`](results/phase4_serving.md) ·
[`results/phase5_drift.md`](results/phase5_drift.md) ·
[`results/phase6_production.md`](results/phase6_production.md)

## Repository layout

```
producer/       replay_producer.py, correctness verification
streaming/      PyFlink velocity aggregator, Kafka connector jar download
feature_store/  Feast entity/feature-view definitions, skew tests
src/driftline/  data loading, leakage-safe splits, baseline model, graph construction, GraphSAGE
scripts/        every experiment/analysis script (baseline, decay curve, drift monitoring,
                drift-triggered retrain, ONNX export, load-test helpers, CI quality gate)
serving/        FastAPI scoring service, Dockerfile, Locust load test
k8s/            k3d Deployment/Service/HPA/PodDisruptionBudget manifests
monitoring/     Prometheus scrape config, Grafana dashboard + provisioning
tests/          unit, leakage, Feast-contract, and testcontainers integration tests
results/        phase-by-phase findings, every number sourced from a real run
metrics/        the consolidated, resume-ready rollup
TASKS.md        the full build log — every checklist item, what's done, what's honestly not
```

## Reproduce it

Everything here ran on a GCP `e2-standard-4` (4 vCPU, 16GB RAM, CPU-only, no GPU) — a modest,
stated machine, not a hidden GPU cluster.

```bash
# 1. Data (needs a Kaggle account + accepted competition rules)
python scripts/download_data.py

# 2. Baseline model (Phase 1) -- ~3 min
python scripts/run_baseline.py
python scripts/random_vs_time_split.py       # the "metrics lied to you" artifact

# 3. Streaming infra (needs Docker) -- bring up Redpanda/Redis/Postgres/MLflow
docker compose up -d
python producer/replay_producer.py --events-per-sec 200
python streaming/velocity_aggregator.py       # needs Java 11 + streaming/download_jars.sh first

# 4. Graph model (Phase 3) -- ~5 min on 4 vCPU
python scripts/train_graphsage.py

# 5. Serving (Phase 4)
python scripts/export_onnx.py
uvicorn serving.app:app --port 8000
python scripts/e2e_smoke_test.py

# 6. Drift + retrain (Phase 5) -- ~15 min, trains XGBoost several times
python scripts/decay_curve.py
python scripts/drift_monitoring.py
python scripts/drift_triggered_retrain.py     # needs MLflow running (part of docker compose)

# 7. Tests
pytest tests/ -v
```

## Runbook

**Consumer lag climbing (Redpanda):** check `docker exec driftline-redpanda rpk group describe
<group>` for per-partition lag. If sustained, the producer's replay rate (`--events-per-sec`)
likely exceeds what the Flink job's neighbor-sampling + Redis/Parquet writes can absorb on this
hardware — reduce the replay rate or add Flink parallelism (`env.set_parallelism`).

**PSI fires (retrain trigger crosses `N_FEATURES_TRIGGER` in `scripts/drift_monitoring.py`):**
this is expected, not an incident — `scripts/drift_triggered_retrain.py`'s loop handles it
automatically: retrain on the window since the last retrain, shadow-score on a held-out tail of
that same window, and only promote if PR-AUC doesn't regress more than `PROMOTION_TOLERANCE`
(0.02) vs. the currently-serving model on that same slice. Check the MLflow run
(`driftline-drift-triggered-retrain` experiment) for the actual shadow-vs-serving numbers behind
the decision.

**Shadow model fails the promotion gate:** by design, the currently-serving model keeps serving
— nothing changes. Check the MLflow run's `regression` metric and the PSI-breached feature list
in `results/drift_monitoring.json` for that week; a rejected promotion usually means the drift
window was too short/noisy to retrain on cleanly (see `MIN_WEEKS_BETWEEN_RETRAINS` in
`scripts/drift_triggered_retrain.py`) rather than a fundamentally bad candidate.

**Scorer pods failing liveness/readiness probes under load:** this is a known, reproduced
finding (see `results/phase6_production.md`) — a single uvicorn worker per pod can't answer
`/health` while saturated handling `/score` requests. Fix: run `uvicorn --workers N` per pod, or
lower the HPA's per-pod CPU target so pods scale out before saturating. Not yet applied here.

## Known Gaps

Logged honestly throughout the build, consolidated: **[`TASKS.md`](TASKS.md)** (bottom section) —
covers scope decisions (Spark alternative path not built, device/email graph entities not added,
GraphSAGE not yet wired into serving, k3d manifests cover the scorer only) and real limitations
found along the way (multi-worker uvicorn fix not yet re-benchmarked, no demo video recorded).
