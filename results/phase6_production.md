# Phase 6 — Production Checklist: results and findings

## CI (GitHub Actions, 5 jobs)
- `lint-and-unit-test`: ruff + data-free tests (leakage, graph, split invariants).
- `feast-contract-test`: Phase 2's deferred schema contract test, now actually running in CI.
- `integration-test`: real testcontainers Redpanda + Redis, 1,000-event produce/consume,
  Redis online-store round-trip, and a scored-event feature-attachment check.
- `onnx-and-quality-gate`: real ONNX export + parity check, real PR-AUC regression gate — both
  against a small (6,000-row, 1.1MB) committed sample, since the full 590K-row dataset needs
  Kaggle credentials and doesn't belong in every CI run. The full-scale numbers everywhere else
  in this repo always come from the complete dataset on the project's VM.
- `docker-build`: builds the actual `serving/Dockerfile` image.

**Two real bugs found and fixed by actually running CI, not assumed:**
1. **Disk exhaustion** (`No space left on device`): installing the full `requirements.txt` in
   every job pulled torch's default CUDA build (nvidia-cublas/cudnn/cufft/etc — several GB a
   CPU-only runner never uses) plus mlflow/evidently/dask/kaggle regardless of whether that job
   needed them. Fixed by giving each job only its own dependencies (`requirements-ci-core.txt` +
   job-specific extras) and installing torch from the CPU wheel index.
2. **Silent numpy upgrade breaking XGBoost**: onnx/onnxruntime declare `numpy>=2` with no upper
   bound; installing them in a separate `pip install` from the pinned `numpy==1.26.4` silently
   upgraded numpy, which breaks `xgboost==2.0.3`'s internal `np.NaN` usage (removed in numpy 2.0).
   **The exact same failure mode hit the local dev machine while testing this session** (torch's
   DLLs briefly broke from the same churn) — documented and fixed the same way in both places:
   pin numpy explicitly in the same install command as the numpy>=2-requiring packages.
- 21 ruff lint errors also surfaced (unused imports, import ordering, blind-except) across scripts
  written this session — fixed via `ruff check --fix --unsafe-fixes` plus one deliberate rule
  ignore (`BLE001`, justified in `pyproject.toml`: this codebase's `except Exception` blocks are
  intentional availability checks, not swallowed bugs).

## k3d production checklist
- Resource limits/requests: set on the scorer Deployment (`cpu: 1`, `memory: 1Gi` limits).
- PodDisruptionBudget: `minAvailable: 1` on the scorer.
- HorizontalPodAutoscaler: **tested by actually driving load, not just applied and assumed
  working.**

### HPA scale event — real, observed
Drove 100 concurrent Locust users against the k3d-deployed scorer for 150s. **A genuine scale
event occurred**: `kubectl` events show
`Scaled up replica set driftline-scorer-7ff86f5854 from 2 to 4` with reason
`cpu resource utilization (percentage of request) above target` — the HPA is not just configured,
it fired correctly under real load.

### A second, more serious finding this load test surfaced: probe starvation under load
Nearly every pod (not just one) hit liveness/readiness probe failures during the test:
`Liveness probe failed: ... context deadline exceeded` and `connection refused`, and one pod was
killed and restarted by kubelet as a direct consequence. **This is the Phase 4 single-uvicorn
-worker finding (p50 latency jumps from 21ms to 240ms under 50-user load due to queueing)
compounding further**: at 100 concurrent users, each pod's single worker became so saturated it
couldn't even answer a trivial `GET /health` within the probe timeout, causing kubelet to treat
healthy-but-busy pods as dead and restart them — a real production reliability risk this
architecture would hit under sustained heavy load, not a hypothetical one.

**A related testing-methodology finding, also worth stating honestly:** the Locust run against
this load showed a 99.71% failure rate (68,809/69,009 requests, mostly `ConnectionRefusedError`).
This number on its own would be misleading to quote as "the scorer's failure rate under load" --
`kubectl port-forward` tunnels to one specific pod, not the Service's load-balanced endpoint set,
so when that pod got killed by its failing liveness probe mid-test, the tunnel broke and every
subsequent request failed for the rest of the run. The real, load-bearing findings are the two
above (a genuine HPA scale event, and genuine probe starvation under load); the raw Locust
failure percentage is a test-harness artifact of port-forwarding to a single pod, not a
scorer-wide production number. Both are recorded because both are real and both are useful --
just not interchangeable.

**Direct fix identified in both Phase 4 and here, not yet applied:** run multiple uvicorn workers
per pod (`--workers N`) so a pod's HTTP server can still answer health checks while other workers
handle scoring requests. Logged in Known Gaps as the concrete next step.

## Leakage regression test
Consolidated at the integration level (`tests/test_leakage_consolidated.py`), re-asserting both
guarantees already tested individually (`test_data.py`'s time-split boundary,
`test_graph.py`'s train-only-graph boundary) using the actual `time_ordered_split` +
`build_graph_edges` functions the real training scripts call, plus a guard against
label-derived feature names.

## What's not done yet (Known Gaps)
- k3d manifests for producer/Flink-job/monitor — **deliberately scoped down**, not an oversight:
  the producer is a batch/replay job and the Flink job would need a full Flink-on-Kubernetes
  operator, both substantial additional infra beyond this build's time budget. Only the scorer
  (the actual long-running production service) has a k8s Deployment.
- Multi-worker uvicorn re-benchmark (the direct fix for the probe-starvation finding above).
- A load-test methodology that doesn't rely on `kubectl port-forward`'s single-pod tunnel (e.g.
  a load generator running inside the cluster, or a NodePort/Ingress) for a genuinely
  representative Service-level throughput number.
