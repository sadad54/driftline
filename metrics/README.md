# Driftline — consolidated metrics

Every number below is sourced from a committed `results/*.json` file (see `metrics/summary.json`
for exact provenance) and reproducible by re-running the referenced script. Hardware: GCP
`e2-standard-4` (4 vCPU, 16GB RAM), CPU-only, no GPU — all numbers stated on this hardware.

## Model quality (time-ordered holdout, 118,108 rows)

| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision@k |
|---|---|---|---|---|
| **XGBoost baseline** | **0.4751** | **0.8892** | **0.3760** | **0.8163** |
| GraphSAGE alone | 0.2452 | 0.8393 | 0.2227 | — |
| XGBoost + GraphSAGE (rank-avg) | 0.4687 | 0.8894 | 0.3765 | — |
| XGBoost + IsolationForest (rank-avg) | 0.4269 | 0.8814 | 0.3501 | 0.7646 |

**Honest finding:** neither ensemble beats the XGBoost baseline alone. Root cause identified and
documented (naive equal-weight rank-averaging, GraphSAGE genuinely undertrained at 5 CPU-only
epochs) — see `results/phase3_graph.md`.

## No-history ("new card") slice — 823 transactions, 0.70% of test set
- Fraud rate: **4.62%** vs. 3.44% overall (**~34% higher** — a genuine independent finding).
- XGBoost alone: PR-AUC 0.6828. Ensemble: 0.6219 (still doesn't beat baseline here either).

## The "metrics lied to you" story
| Split | PR-AUC | Recall@1%FPR |
|---|---|---|
| Time-ordered (honest) | 0.4751 | 0.3760 |
| Random 80/20 | 0.6974 | 0.6175 |
| **Inflation** | **+0.2222** | **+0.2415** |

## Drift: performance decay without retraining
| Month | PR-AUC |
|---|---|
| 1 (in-sample — training data itself, not a fair "decay" endpoint) | 0.9069 |
| **2 (first genuinely held-out month)** | **0.4761** |
| 6 | **0.3680** |

Honest decay: **month 2 → month 6, -22.7% relative**, no retraining.

## Drift monitoring (369 features, weekly, manual PSI+KS)
- First retrain trigger: **week 2** (30 features PSI > 0.2) — before month 1 even ends.
- Persistent breach: 16-35 features/week for the rest of the replay.

## Drift-triggered retraining: shadow-scored, gated, recovered
- **2/2 triggered retrains passed the causal shadow-scored promotion gate.**
- **The recovered delta:** week 4 PR-AUC 0.5913 degraded to week 14's **0.2635** (worse than the
  untreated month-6 baseline) → retraining recovered it to week 15's **0.5211** — **+0.2576
  absolute, +97.8% relative, in a single week.**

## Serving
- ONNX parity (native XGBoost vs. ONNX Runtime): max abs diff **2.98e-7**.
- Single-request latency: **20.98ms**.
- Load test (50 users, docker-compose): **10,018 requests, 0 failures**, throughput **170-200
  req/s**, **p50 240ms / p95 380ms / p99 490ms**. p50 under load vs. 21ms single-request is
  queueing (single uvicorn worker), not slower inference — direct fix identified, not yet applied.
- k3d HPA: **real scale event observed** (2→4 replicas, CPU-triggered). Same load test surfaced
  near-simultaneous liveness-probe failures across pods and a kubelet-triggered restart — a real
  production risk from the same single-worker root cause above, now observed at the cluster level.

## Analyst-budget precision@k (42-day test period, ~2,812 transactions/day mean)
| Budget | Reviews/day | Mean precision@k |
|---|---|---|
| 0.5% | ~14 | **87.7%** |
| 1.0% | ~28 | **75.6%** |
| 2.0% (≈"200 analysts/10,000 flags") | ~56 | **59.6%** |
| 5.0% | ~141 | **37.1%** |

## Training-serving skew
- Feast materialize vs. source: 0/1,500 mismatches (trivially consistent, can't structurally diverge).
- **Real-time Redis vs. buffered Parquet: 240/2,772 mismatches (8.66%)** — root cause: Parquet
  sink buffers 200 rows before flush, Redis sink writes per-event, so online is briefly *ahead*
  of offline (the reverse of the usual assumption).

## Feature freshness lag (produced_at → online-store write)
| Window | p50 | p95 | p99 | max |
|---|---|---|---|---|
| 1h | 14.7s | 23.9s | 25.6s | 27.1s |
| 24h | 14.1s | 27.9s | 32.7s | 37.0s |
| 7d | 7.4s | 25.7s | 34.9s | 37.2s |

## CI quality gate
`onnx-and-quality-gate` job in GitHub Actions: trains XGBoost on a committed 6,000-row sample,
fails the build if PR-AUC regresses more than **0.02 absolute** vs. a checked-in baseline
(widened from 0.01 after a real run showed that tolerance was too tight for genuine cross-machine
XGBoost threading nondeterminism — see `results/phase6_production.md`). **All 5 CI jobs
(lint/unit, Feast contract, testcontainers integration, ONNX+quality-gate, docker build) are
green** as of the last push to `main`.
