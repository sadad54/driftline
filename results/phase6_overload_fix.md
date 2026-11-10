# Phase 6 follow-up: probe starvation under load

**Status: fix implemented and tested locally. The Kubernetes load test has NOT been re-run yet.**
The Phase 6 numbers in `phase6_production.md` (HPA 2→4, liveness failures, one restart) still
describe the code before this change. The cluster re-run is pending; the procedure is in
[`docs/overload-rerun.md`](../docs/overload-rerun.md).

## The finding being fixed

In the Phase 6 load test, scorer pods failed liveness probes while under load and kubelet
restarted one of them (see `phase6_production.md`). A pod that is busy but healthy was treated as
dead, which removes capacity at the moment it is needed most.

## What changed

| Area | Before | After |
|---|---|---|
| Health endpoints | One `/health`, used for both probes | `/healthz` (liveness: process responsive) and `/readyz` (readiness: model loaded). `/health` kept for existing scripts |
| Probe settings | Defaults: 1 s timeout, 3 failures, so ~30 s of slow answers kills the pod | Liveness: 3 s timeout, 6 failures (60 s of silence). Readiness: 2 s, 3 failures. Startup probe covers model load |
| Admission control | None. Every request is accepted and queued | At most `DRIFTLINE_MAX_INFLIGHT` (16) `/score` requests admitted. The rest get `503` + `Retry-After: 1` immediately, before the body is read |
| Where scoring runs | Starlette's shared 40-thread pool | A dedicated pool of `DRIFTLINE_SCORER_THREADS` (2); the event loop only parses requests and answers probes |
| ONNX Runtime threads | Default: one spinning thread per host core, inside a 1-CPU limit | 1 intra-op thread |
| Metrics | none for overload | `driftline_score_shed_total`, `driftline_score_inflight` |

Two things were considered and rejected:

- **`uvicorn --limit-concurrency`.** It sheds load cheaply, but it applies to every connection,
  including the probe's. Under overload the liveness probe would get the 503.
- **Failing readiness when saturated.** The pod already refuses work it cannot do. Removing it
  from the Service as well moves its traffic to the other pods and pushes them over too.
  `/readyz` reports `saturated` in its body and stays 200.

## Local measurement

`scripts/probe_starvation_bench.py` runs one scorer process under a CFS CPU quota (the mechanism
behind Kubernetes `limits.cpu`), saturates it with a Go client, and probes its health endpoint
every 0.5 s with a 1 s timeout, as kubelet does by default. "Before" is the commit preceding this
change (`f350b3b`), "after" is this branch. Each run is 30 s. Raw output is in `results/overload/`.

| CPU quota | Clients | Version | Probes failed | Slowest probe | Score p50 / p99 | Scored OK | Shed (503) |
|---|---|---|---|---|---|---|---|
| 1.0 | 100 | before | 0 / 60 | 79 ms | 85 / 134 ms | 34,258 | 0 |
| 1.0 | 100 | after | 0 / 60 | 31 ms | 12 / 26 ms | 40,666 | 45,609 |
| 1.0 | 500 | before | 0 / 60 | 311 ms | 464 / 617 ms | 32,192 | 0 |
| 1.0 | 500 | after | 0 / 60 | 42 ms | 26 / 65 ms | 23,772 | 224,926 |
| 0.25 | 100 | before | 0 / 60 | 380 ms | 401 / 606 ms | 7,293 | 0 |
| 0.25 | 100 | after | 0 / 60 | 103 ms | 88 / 188 ms | 8,100 | 24,847 |
| 0.25 | 500 | before (3 runs) | **18/35, 19/35, 22/33** | 1 s timeout | ~2,100 / 7,700-9,600 ms | ~7,100 | 0 |
| 0.25 | 500 | after (3 runs) | **0/60, 0/60, 0/60** | 324-395 ms | 244-265 / 1,500-2,500 ms | ~2,600 | ~77,000 |

In the failing "before" runs the longest streak of consecutive failed probes was 16, 10 and 11.
Kubelet's default restarts a container after 3.

### What these numbers do and do not show

- **The failure reproduces only when the pod is starved well below its limit.** With a full
  1-CPU quota the old code never missed a probe, even at 500 clients (slowest answer 311 ms).
  It failed at a quarter of a core with 500 clients. The pod spec has `requests.cpu: 250m` and
  `limits.cpu: 1`, and the Phase 6 cluster, the load generator and everything else shared one
  VM. On an oversubscribed node a pod is only guaranteed its request. So the Phase 6 restarts
  are consistent with **node-level CPU starvation plus the default 1 s / 3-failure probe**, not
  only with "a single uvicorn worker is too slow". This is a hypothesis from a local reproduction;
  the re-run is designed to confirm or reject it.
- **Under the reproducing condition the fix holds:** 0 failed probes in 3 runs against 18-22.
- **Admitted requests get much faster** (p99 617 ms → 65 ms at 1 CPU / 500 clients) because the
  queue is bounded.
- **Shedding is not free.** At 500 clients the fixed version completes fewer requests in total
  (23,772 vs 32,192 at 1 CPU; ~2,600 vs ~7,100 at 0.25 CPU). Refusing ~7,500 requests a second
  costs CPU, and the test client retries 50 ms after every 503, which is more aggressive than
  a client that honours `Retry-After: 1`. The trade is deliberate: fewer requests, each answered
  quickly, and a pod that stays alive, against more requests that each take seconds and a pod
  that gets restarted.
- **Limits of the setup.** 2-core sandbox, model trained on the 6,000-row CI sample (smaller and
  faster than the full model), Feast not connected, one process, no kubelet, no HPA. The absolute
  numbers will differ on the cluster. The comparison between versions is the point.

## Tests

`tests/test_serving_overload.py` (7 tests, run in CI) starts the real app on a real uvicorn
server with a fake model that blocks until released, so "every scorer thread is busy" is a state
the test can hold and inspect:

- liveness answers in under 0.5 s while all admitted requests are stuck in the model
- a request beyond the limit gets 503 + `Retry-After` immediately and never reaches the model
- readiness stays 200 when saturated and reports it; it fails only while the model is not loaded
- capacity returns after a burst, and a scoring exception does not leak a slot
- shed requests show up in `/metrics`

## Still open

- Re-run the Kubernetes load test (`docs/overload-rerun.md`) and replace the "pending" status above.
- HPA target is unchanged at 70% CPU. Whether it should scale earlier is a question for the re-run.
