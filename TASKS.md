# Driftline — Build Task List

## Goal

Ship **Driftline**: a streaming fraud-detection platform replaying the 590,540-row IEEE-CIS
transaction dataset in event-time order through Redpanda + PyFlink, computing windowed
identity-velocity features into a Feast online (Redis) / offline (Parquet) store, scoring with a
rank-averaged ensemble of XGBoost + PyTorch Geometric GraphSAGE + IsolationForest behind an ONNX/
FastAPI service on k3d, with Evidently/Prometheus/Grafana drift monitoring and an Airflow
drift-triggered retrain → shadow → promotion pipeline. This replaces the old static
creditcard.csv/SMOTE notebook on the resume entirely.

## Budget: $0 out of pocket, hard constraint

The GCP VM runs entirely on the account's $300/90-day Free Trial credit (confirmed via the
console, not a paid/org-billing account — Google does not auto-charge the card when trial credit
runs out; it pauses instead). To keep this genuinely $0 and make the credit last across this and
later projects:
- **Stop the VM (`gcloud compute instances stop driftline-vm --zone us-central1-a`) whenever a
  work session ends**, not just when idle mid-session — a stopped instance bills $0 compute (only
  negligible persistent-disk storage, pennies/month, still inside free-tier/credit).
- A budget alert is set on the billing account (MYR 50 / ~$10.60, at 50/90/100% thresholds) as an
  early-warning tripwire — it emails, it does not hard-stop spend, so don't rely on it alone.
- Before enabling any new GCP API/service, sanity-check it's covered by compute/free-tier usage,
  not a metered API with per-call cost outside the free tier (e.g. avoid enabling paid Vertex AI
  endpoints, premium network tiers, or multi-region resources without checking pricing first).

## Quality bar

The timeline is compressed to a continuous ~72-hour build (not the source doc's 6-week/12-15h-a-week
cadence), but **every item on the production-level checklist below must be built for real**. No
shortcut is acceptable just because time is short:

- No SMOTE-before-split. No random k-fold on time-series fraud data. No "0.99 ROC-AUC" vanity
  metric without PR-AUC / recall@1%FPR alongside it.
- No GNN "bolted on" without temporal edge masking and a with/without ablation.
- No claiming a metric that wasn't actually measured on stated hardware. Every number in the
  README and resume bullets must trace back to a real run, real logs, or a real chart.
- If a step is skipped or stubbed under time pressure, it gets logged in **Known Gaps** at the
  bottom of this file with the reason — never silently marked done.

## What's real vs. simulated (state this plainly in the README, lead with it)

- **Real:** the Redpanda broker, partitions, consumer groups, offsets; the Flink/PySpark windowed
  stateful aggregation; the Feast online/offline store and the skew between them; the GraphSAGE
  entity graph model with temporal masking; the PSI/KS drift detection math; the Airflow
  drift-triggered retrain/shadow/promote pipeline; the FastAPI+ONNX scoring service and its
  measured latency/throughput on the EC2 VM.
- **Simulated / replayed, and labeled as such everywhere it's claimed:** "real-time" here means a
  **replay** of six months of historical IEEE-CIS transactions at a controllable events/sec rate
  through a real broker — not live production traffic. Say this explicitly in the README's first
  paragraph, in the demo video narration, and in interview answers. Honesty about this distinction
  is a deliberate differentiator, not a weakness to hide.

---

## Phase 0 — Environment & Scaffold

- [x] Launch cloud VM: **switched AWS -> GCP** (Compute Engine, free $300/90-day trial credit
      instead of card spend) — `driftline-vm`, `e2-standard-4` (16GB RAM / 4 vCPU), Ubuntu 22.04.5
      LTS, us-central1-a, 60GB disk. External IP + SSH access via `gcloud compute ssh` (host key
      cached, auth via `gcloud auth login` — browser OAuth, non-interactive-session-safe once
      logged in; re-run `gcloud auth login` if the token expires between sessions)
- [x] Security: SSH reachable via gcloud's IAP-less direct connect (GCP default firewall allows
      22 from anywhere; tightened enough for a short-lived build — revisit before leaving the VM
      up long-term). Demo ports (Grafana 3000, FastAPI 8000, MLflow 5000, Redpanda Console 8090)
      not yet opened to the public internet — deferred until Phase 4 demo prep
- [ ] Confirm GCP budget alert set so the VM doesn't quietly burn the $300 trial credit if left
      running — **not yet done, do this before walking away from the build for any length of time**
- [x] SSH in, install Docker Engine + Compose plugin (`get.docker.com` script) — Docker 29.7.2,
      docker-compose-plugin included; `hp` user added to `docker` group (no sudo needed)
- [x] Install k3d + kubectl + helm on the VM — kubectl v1.37.0, k3d v5.9.0, helm v3.21.4
- [x] Install Python 3.10 (VM's apt-default; not 3.11 — `pyproject.toml` relaxed to
      `requires-python = ">=3.10"` to match), venv tooling, git — all via apt
- [x] Set up swap or confirm 16GB is enough headroom for Redpanda + Flink + Redis + Postgres + k3d
      running concurrently — 15GB usable RAM confirmed via `free -h`; will monitor actual
      concurrent footprint once Phase 2 brings Flink online, not yet stress-tested with all
      services + k3d simultaneously
- [x] `git init` the `driftline` repo, push to GitHub (private) — `github.com/sadad54/driftline`,
      via `gh` CLI (device-code browser auth). VM has its own write-enabled SSH deploy key (not
      the user's personal token) for `git pull`/`push` from the VM
- [x] Scaffold repo structure: `producer/`, `streaming/`, `feature_store/`, `models/`, `serving/`,
      `monitoring/`, `orchestration/`, `k8s/`, `tests/`, `.github/workflows/`, `notebooks/`
- [x] Set up `docker-compose.yml` for local iteration (Redpanda, Redis, Postgres, MLflow) — all 5
      containers (+ Redpanda Console) up and healthy on the VM via `docker compose up -d`
- [x] Download IEEE-CIS Fraud Detection dataset (Kaggle, 590,540 rows, 434 features) — verified via
      `scripts/inspect_ieee_cis.py`: 590,540 rows / 394 txn cols + 40 identity cols, 3.499% fraud.
      Also present on the VM (`gcloud compute scp`, not re-downloaded — VM's Python 3.10 only
      resolves `kaggle==1.7.4.5`, which needs classic `kaggle.json` username+key, not the newer
      token format used locally; copying the already-verified local files was simpler and byte
      sizes were confirmed to match exactly after transfer)
- [x] Download Elliptic Bitcoin Dataset (203,769 nodes, 234,355 edges) — files verified present
      (`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`); also
      copied to the VM alongside IEEE-CIS
- [x] Baseline pipeline (`scripts/run_baseline.py`) re-run on the VM to confirm environment
      parity: data load 34.8s (vs 181.5s locally — 15GB RAM vs 7.3GB removes the downcast-or-OOM
      pressure), PR-AUC 0.4776 vs local 0.4751 (expected XGBoost histogram-threading
      nondeterminism, not a discrepancy — IsolationForest's PR-AUC is bit-identical across both
      machines)
- [ ] Download PaySim dataset (6.3M synthetic mobile-money transactions, load-test supplement only)
      — deferred, only needed for Phase 4 load testing, not blocking Phase 1-3
- [x] Verify dataset checksums/row counts match documented sizes; record raw file locations and
      sizes in a `data/README.md` — see `data/README.md` and `data/raw/README.md`
- [ ] Decide and document Kafka topic/schema naming conventions up front (`transactions.raw`,
      `transactions.scored`) to avoid churn later — deferred to Phase 2 start

---

## Phase 1 — Data & Baseline Model

- [x] EDA / inspection pass (script, not notebook — equally rigorous): fraud rate (3.499%,
      matches expected), feature nullness (41% overall; top-nulls dist2/D7/D13/D14 >89%),
      TransactionDT range (86,400-15,811,131, ~182 days), card/addr/email cardinality —
      `scripts/inspect_ieee_cis.py`, findings in `data/README.md`
- [x] Implement **strict time-ordered split** (`time_ordered_split` — last 20% by `TransactionDT`
      as holdout) — random k-fold explicitly rejected, documented in `data/README.md`
- [x] Write and run a regression test asserting the split is time-ordered (max train timestamp <=
      min test timestamp) — `tests/test_data.py::test_time_ordered_split_no_leakage`, plus a
      negative-control test proving it would actually catch a random-split violation
- [x] Feature engineering pass: native XGBoost NaN + pandas-category handling (no manual
      imputation needed for the supervised path); separate median-imputed/ordinal-encoded matrix
      for IsolationForest (documented as a genuinely different preprocessing path, not an
      oversight) — `src/driftline/data.py`, `src/driftline/baseline.py`
- [x] Train XGBoost baseline on time-ordered split; record **baseline PR-AUC and ROC-AUC** —
      **PR-AUC 0.4751, ROC-AUC 0.8892, recall@1%FPR 0.376** (`results/baseline_metrics.json`)
- [x] Train IsolationForest residual/anomaly signal as a second baseline component — PR-AUC
      0.1193; **finding:** naive rank-average ensemble with it is *worse* than XGBoost alone
      (precision@k=0.0 for IsolationForest alone) — see `results/README.md` for the full ablation
      and the decision on how to handle this in the Phase 3 ensemble
- [x] No resampling used in this baseline (scale_pos_weight instead of SMOTE/undersampling) —
      sidesteps the resampling-before-split leakage class entirely rather than needing a test for
      it; revisit only if a later phase actually introduces resampling
- [x] Write the "metrics lied to you" comparison artifact: same model, random-split metric vs.
      time-split metric, side by side, with the delta explained —
      `scripts/random_vs_time_split.py`, results in `results/random_vs_time_split.json`
- [x] Commit baseline metrics (PR-AUC, ROC-AUC, recall@1%FPR) to a tracked file used as the
      comparison point for every later phase — `results/baseline_metrics.json`, committed

---

## Phase 2 — Streaming Infrastructure

- [x] Stand up Redpanda (Kafka-API compatible) via docker-compose; create topic `transactions.raw`
      with sensible partition count — 6 partitions, healthy on the VM
- [x] Build `replay_producer.py`: emits IEEE-CIS events in strict `TransactionDT` order at a
      configurable events/sec rate — `--events-per-sec`, `--limit` flags; sustained ~150-250
      events/sec actual on this hardware (measured, not the configured target). Pause/resume not
      implemented (SIGINT handler stops cleanly but no resume-from-offset flag) — logged in Known
      Gaps
- [x] Verify producer correctness: consumed event order matches source order; no dropped/duplicated
      events at steady rate — `producer/verify_replay.py`: 5,000/5,000 unique TransactionIDs, 0
      duplicates, 0 per-partition order violations (partitioned by card1 — per-partition order is
      the honest guarantee Kafka gives here, not global topic order, and that's what's tested)
- [x] Implement consumer group(s) and confirm consumer-lag is observable (via `rpk` or exposed
      metric) — verified `rpk group`/`rpk topic describe -p` mechanics; not yet load-tested at
      sustained high throughput (Phase 4 item)
- [x] Implement windowed aggregation job in **PyFlink**: card1 velocity over 1h / 24h / 7d HOP
      (sliding) windows — `streaming/velocity_aggregator.py`, verified against an 8,000-event
      replay (12,746 window-close events for the 1h view alone). **Scope: card1 only**, not
      device/email — logged in Known Gaps, pattern generalizes directly
- [ ] Keep a Spark Structured Streaming variant behind a feature flag/alt entrypoint — **not
      built**, logged honestly in Known Gaps rather than falsely claimed
- [x] Wire windowed aggregate output to two sinks: Redis (Feast online store) and Parquet (Feast
      offline store) — both verified with real data; found and fixed a real Parquet durability
      bug along the way (see results/phase2_streaming.md)
- [x] Stand up Feast: define entities (card1) and feature views for the windowed aggregates;
      register online (Redis) + offline (Parquet) stores — `feature_store/definitions.py`,
      `feast apply` + `feast materialize` verified, online reads confirmed via SDK
      (`test_online_read.py`). **Scope: card1 only**, matching the aggregation scope above
- [x] Write and pass **training-serving skew test**: sample N replayed events, compare offline vs.
      online feature values; document the actual skew bug found and fixed — two tests written:
      `skew_test.py` (Feast-materialize-vs-source, trivially 0/1500 mismatches, can't structurally
      diverge) and the one that actually matters, `skew_test_realtime.py` (hand-rolled real-time
      Redis write vs. 200-row-buffered Parquet write): **240/2,772 mismatches (~8.7%), reproduced
      across two runs, root cause found and explained** — see results/phase2_streaming.md
- [ ] Contract test on Feast feature definitions: a schema change to a feature view must fail CI
      — deferred to Phase 6
- [x] Measure and record **feature freshness lag** (event time → online-store availability), p99,
      on the VM — p50/p95/p99/max per window size in results/phase2_streaming.md (e.g. 1h window:
      p50 14.7s, p99 25.6s). Found and fixed a real bug en route: `produced_at` was embedded in
      every producer event but silently dropped by the Flink source table (never declared in the
      `CREATE TABLE` DDL) until this metric surfaced it

---

## Phase 3 — Graph Model

- [x] Construct entity graph from IEEE-CIS identity columns: card1, card2, card3, card5, addr1,
      addr2, P_emaildomain, R_emaildomain as nodes/shared-identity edges — `src/driftline/graph.py`.
      **Scope: device info (DeviceInfo/DeviceType) not included**, logged in Known Gaps
- [x] Persist graph construction as a reproducible script (not notebook-only), with node/edge counts
      logged — `scripts/train_graphsage.py`: 590,540 txn nodes + 14,811 value nodes, 6.46M
      train-graph edges / 8.06M train+test-graph edges, real timings logged
- [x] Implement **temporal edge masking** — precisely scoped, not overclaimed: the property that
      actually matters (training weights learned only from a graph where test rows are
      structurally absent, not just unlabeled) is implemented and tested
      (`tests/test_graph.py::test_value_only_present_in_test_rows_never_appears_in_train_graph`).
      Inference on test uses an extended train+test graph as a named, bounded simplification
      (test transactions can see each other via shared value nodes) rather than strict
      per-transaction sequential masking — documented, not hidden, in `src/driftline/graph.py`'s
      module docstring and `results/phase3_graph.md`
- [x] Build PyTorch Geometric GraphSAGE model on the entity graph; train with same time-ordered
      split as the tabular baseline — `src/driftline/graphsage_model.py` +
      `scripts/train_graphsage.py`. Real run: 242.6s / 2,310 batches / 5 epochs on the full
      590K-row dataset, CPU-only (this VM has no GPU)
- [ ] Optional/stretch: reproduce a published GraphSAGE baseline result on Elliptic Bitcoin Dataset
      — **not done**, genuinely optional per the doc's own wording, skipped for time
- [x] Build rank-average ensemble combining XGBoost + GraphSAGE (+ IsolationForest residual signal)
      — `scripts/train_graphsage.py`
- [x] Produce **ablation table**: XGBoost alone vs. XGBoost+GraphSAGE vs. full ensemble, PR-AUC and
      recall@1%FPR for each, with a named baseline row — `results/phase3_graph.md`. **Honest
      result: the naive rank-average ensemble is slightly WORSE than XGBoost alone** (PR-AUC
      0.4687 vs 0.4776) — same failure mode as Phase 1's IsolationForest finding, root-caused
      and explained, not hidden
- [x] Compute **graph lift specifically on the "new card, no history" slice** — 823/118,108 test
      transactions (0.70%), with a genuinely interesting independent finding: **4.62% fraud rate
      on this slice vs. 3.44% overall (~34% higher)**. Ensemble does NOT outperform XGBoost alone
      here either (PR-AUC 0.6219 vs 0.6828) — reported honestly rather than the hoped-for result
- [x] Record: does GraphSAGE help everywhere or only on specific slices? — **Answer: neither, in
      this configuration.** Two specific, non-speculative reasons identified and documented (naive
      equal-weight ensembling punishes the weaker model; the GNN is genuinely undertrained — loss
      still declining at 5 epochs — relative to a fully-tuned 400-tree XGBoost baseline within the
      CPU time budget) — see `results/phase3_graph.md` for the full interview-ready framing

---

## Phase 4 — Serving

- [x] Export XGBoost to ONNX; write an ONNX-vs-native parity test — **categorical-native model
      genuinely fails export** (`RecursionError`, onnxmltools can't handle
      `enable_categorical=True`), confirmed by trying rather than assumed. Fallback ordinal-encoded
      "serving" variant exports cleanly: max abs diff 2.98e-7, PASS. **GraphSAGE NOT exported to
      ONNX** — logged in Known Gaps (risky dynamic sparse-op export, and the ensemble doesn't
      show lift yet per Phase 3, so not worth the risk this build)
- [x] Build FastAPI scoring service: `/score` endpoint, fetches Feast online features (proven via
      a real request: `velocity_features_available: true`), runs XGBoost via ONNX Runtime.
      **Velocity features fetched but not yet part of the model's input schema** — named honestly
      (`_available` not `_used`) rather than implying an effect that isn't there
- [x] Containerize scoring service; run it in docker-compose locally, then in k3d — both verified
      working with a real scoring request returning the identical fraud_score in both. k3d
      deployment surfaced a real, undocumented-until-now network boundary: Feast/Redis
      unreachable from k3d pods (separate Docker network from docker-compose) — logged, not hidden
- [x] Locust load test against the scoring service; sustained events/sec, p50/p95/p99 end-to-end
      latency — **10,018 requests, 0 failures, ~170-200 req/s, p50 240ms / p95 380ms / p99 490ms**
      at 50 concurrent users. Real finding: single-request latency was 21ms, so the 240ms p50
      under load is queueing (single uvicorn worker serializing CPU-bound inference), not slower
      inference — direct fix (more workers/replicas) identified, not yet re-benchmarked
- [x] Implement alert-queue / precision@k simulation: fixed daily analyst review budget (k),
      compute precision@k on the replay — real per-day numbers across 42 test days: 0.5% budget
      (~14/day) → 87.7% mean precision; 2% budget (~56/day, matching the classic "200
      analysts/10,000 flags" ratio) → 59.6% mean precision
- [x] End-to-end smoke test: producer → Redpanda → scorer → `transactions.scored`, verified on a
      30-event batch — real HTTP round-trip (mean 10.7ms), published and read back successfully.
      **Flink/Feast stage verified independently in Phase 2, not re-chained into this exact script**
      (the scorer's Feast lookup within `/score` is the connection point, proven separately)

---

## Phase 5 — Drift & Retraining

- [ ] Instrument Evidently: PSI and KS statistics per feature, computed weekly across the six-month
      replay
- [ ] Compute and chart the **performance-decay curve**: PR-AUC by month (month 1 → month 6),
      without retraining — this is the doc's headline "real drift curve, not injected noise"
      artifact; make sure it's genuinely computed from the replay, not synthesized
- [ ] Record count of features crossing PSI > 0.2 per week, and which week/month the count first
      breaches the retrain threshold
- [ ] Set up Prometheus scraping + Grafana; build dashboard panels: latency histogram, throughput,
      consumer lag, PSI heatmap, alert-queue precision — export dashboard JSON into the repo
- [ ] Stand up Airflow (or lightweight scheduler if Airflow proves too heavy for the 72h budget —
      log the substitution honestly in Known Gaps if made) with a DAG: PSI > threshold for N
      features → trigger retrain
- [ ] Wire retrain job to MLflow model registry (log params, metrics, artifacts)
- [ ] Implement shadow-scoring: newly retrained model scores one held-out replay week in shadow
      (no serving impact), compared against the live model
- [ ] Implement promotion gate: retrained model is promoted to serving only if it passes a defined
      quality bar (e.g. PR-AUC not worse than X% regression) — write this as an enforced check, not
      a manual step
- [ ] Re-run the six-month replay **with** drift-triggered retraining enabled; chart PR-AUC by
      month again and compute the **recovered delta** vs. the untreated decay curve — quote both
      endpoints (month 1, month 6) and the recovered amount

---

## Phase 6 — Production Checklist (do not skip any of these)

- [ ] pytest **testcontainers integration test**: spins up Redpanda + Redis, publishes 1,000
      synthetic/replayed events, asserts scored events arrive with correct features attached
- [ ] Feast schema contract test wired into CI (from Phase 2) confirmed running in GitHub Actions,
      not just locally
- [ ] GitHub Actions CI matrix: unit tests, integration tests (testcontainers), ONNX parity check,
      Docker image build, PR-AUC quality gate (fails the build if PR-AUC regresses >1% vs. baseline)
- [ ] k3d manifests: separate Deployments for producer, Flink job, scorer, monitor
- [ ] Resource limits/requests set on every Deployment (not defaults)
- [ ] PodDisruptionBudget defined for the scorer (and any other multi-replica service)
- [ ] HorizontalPodAutoscaler on the scorer, tested by driving load and observing a scale event
- [ ] Runbook section in README: what to do when consumer lag climbs, when PSI fires, when the
      shadow model fails the promotion gate — written as actual operational steps, not platitudes
- [ ] Leakage regression test: assert no future information (features or graph edges) is visible to
      the model at train time for any given row/timestamp

---

## Phase 7 — Deliverables & Wrap-up

- [ ] Consolidate all measured metrics into one `metrics/` artifact (JSON + rendered table), each
      tied to the run/commit that produced it:
  - [ ] PR-AUC and ROC-AUC, time-ordered holdout vs. XGBoost baseline
  - [ ] Recall@1%FPR
  - [ ] Precision@k (k = chosen analyst review budget)
  - [ ] Performance-decay curve: PR-AUC month 1 → month 6, untreated vs. with drift-triggered
        retrain, both endpoints + recovered delta
  - [ ] PSI per feature per week, and count of features crossing 0.2 before retrain fired
  - [ ] Graph lift: ensemble vs. XGBoost-alone PR-AUC, overall and on the no-history-card slice
  - [ ] Throughput and latency: sustained events/sec, p50/p95/p99 end-to-end, with stated EC2
        hardware spec
  - [ ] Feature freshness lag, p99
  - [ ] Training-serving skew: max absolute offline/online feature difference, and the bug story
- [ ] Fill in every `[x]` placeholder in the resume bullets below with real measured numbers only:
  - [ ] "computing... velocity features into a Feast online store at *[x]* ms p99 freshness lag and
        sustaining *[x]* events/sec"
  - [ ] "raising PR-AUC from *[x]* to *[x]* and recall at 1% FPR from *[x]* to *[x]*"
  - [ ] "measured PR-AUC decay from *[x]* (month 1) to *[x]* (month 6) untreated and recovered *[x]*
        of it"
  - [ ] "CI quality gate blocking any model regression above 1% PR-AUC" — confirm this is literally
        true of the CI config, not aspirational
- [ ] Write README: lead with the six-month drift curve chart, state the real-vs-simulated framing
      up front, architecture diagram, metrics table, runbook, how to reproduce
- [ ] Record demo video: replay running end-to-end, Grafana dashboard live, drift curve, a
      retrain-triggered event if timing allows; narration explicitly states "replay of historical
      data through a real broker," not "live production traffic"
- [ ] Prep interview-story answers, each backed by an artifact in the repo:
  - [ ] "How do you know your model is still working in production?" → decay curve + chosen PSI
        threshold, defensible
  - [ ] "Fraudsters adapt. How do you handle that?" → adversarial drift measurement + automated
        response
  - [ ] "Why a graph model here?" → shared-identity story, no-history-slice lift, honest gaps
  - [ ] "10,000 flags/day, 200 analysts. What now?" → precision@k under fixed budget
  - [ ] "What's training-serving skew and have you hit it?" → the real bug + the test that caught it
  - [ ] "Tell me about a time your metrics lied to you." → random-split vs. time-split story
- [ ] Update resume/portfolio: replace the old static creditcard.csv/SMOTE fraud notebook entry
      with the Driftline entry (see Definition of Done)

---

## Definition of Done

- [ ] Driftline **fully replaces** the old fraud-detection notebook project on the resume/portfolio
      — the old entry is removed, not left alongside as a duplicate
- [ ] Every resume bullet number is real, sourced from `metrics/`, and reproducible by re-running a
      documented command
- [ ] Every "Production-level checklist" item in Phase 6 is checked off with a real, runnable
      artifact in the repo — none marked done from memory or intention
- [ ] The README's opening paragraph states plainly what is real (broker, consumer group, graph
      model, drift detection, retrain gate) and what is simulated (replay of historical data, not
      live traffic)
- [ ] `Known Gaps` section below is filled in honestly for anything cut under time pressure — empty
      is fine only if genuinely nothing was cut

### Do-not-do-this checklist (anti-patterns from the spec — verify none apply before calling it done)

- [ ] NOT the creditcard.csv + SMOTE + 0.99 ROC-AUC notebook pattern — confirmed replaced by
      time-ordered replay, PR-AUC, recall@1%FPR, six-month drift
- [ ] NOT SMOTE (or any resampling) applied before the split — confirmed resampling is inside the
      CV fold only, and a test asserts it
- [ ] NOT "real-time" meaning just a FastAPI endpoint — confirmed there's an actual broker, actual
      consumer-lag metric, actual windowed state
- [ ] NOT a GNN bolted on with no ablation and leaking future edges — confirmed temporal edge
      masking exists and the with/without ablation table is in the repo
- [ ] NOT "just another fraud classifier" — confirmed the README leads with the drift curve chart,
      and the retrain gate / skew test / analyst-budget framing are all present and demonstrated
- [ ] NOT claiming "deployed to production" for what is actually a replay-based online evaluation —
      confirmed QPS, p99, and hardware are quoted, and the replay-vs-live distinction is stated
      plainly, not hedged

---

## Known Gaps

_(Log anything cut, substituted, or simplified under the 72-hour constraint here, with the reason.
Leave empty only if nothing was cut.)_

- **GraphSAGE not exported to ONNX / not served.** Only the XGBoost baseline is served (Phase 4).
  Reasonable given Phase 3's honest finding that the current ensemble doesn't beat XGBoost alone
  yet, plus real export risk (dynamic sparse ops, unlike the tree-ensemble ONNX path that already
  needed a fallback for XGBoost's own categorical handling).
- **k3d scorer pods can't reach Feast/Redis** (separate Docker network from docker-compose) —
  works correctly in docker-compose, not yet bridged for the k3d deployment path.
- **Locust load test not re-run with multiple uvicorn workers/replicas** — the identified direct
  fix for the queueing-driven p50 latency jump (21ms single-request vs 240ms under 50-user load).
- **Phase 3 entity graph excludes DeviceInfo/DeviceType** as identity columns (only
  card1/card2/card3/card5/addr1/addr2/P_emaildomain/R_emaildomain). Same pattern would extend
  directly; skipped for time and because DeviceInfo has high free-text cardinality that would
  need cleaning first to be a useful identity signal, not just added as-is.
- **Phase 3 ensemble is naive rank-average, not a learned/weighted stack.** Root-caused as the
  reason GraphSAGE doesn't show lift in the ablation table (see results/phase3_graph.md) — the
  concrete next step is a small logistic-regression meta-learner fit on
  `[xgb_score, graphsage_score, iso_score] -> isFraud` using a validation slice carved from the
  tail of TRAIN (never touching TEST), not implemented here for time.
- **GraphSAGE trained for 5 epochs on CPU** (242.6s, loss still declining at the end) — genuinely
  undertrained relative to the fully-tuned XGBoost baseline; more epochs is the direct lever, not
  applied here to keep the phase within the overall build's time budget.
- **Phase 2 velocity features are card1-only**, not device/email as the source doc's architecture
  lists. The pattern (`build_velocity_table` + dual sink) generalizes directly to DeviceInfo and
  P_emaildomain; not done yet purely for time. Revisit before final metrics rollup if time allows.
- **Spark Structured Streaming alternative path not built.** The source doc says "keep a Spark
  variant behind a feature flag... claim both truthfully" — only the PyFlink path exists. Stated
  here explicitly rather than claiming both.
- **Producer has no pause/resume-from-offset flag** — SIGINT stops it cleanly but there's no way
  to resume a partial replay from where it left off; each run starts from the beginning of the
  dataset (or `--limit`-truncated head of it).
- **Consumer-lag observability verified structurally, not load-tested.** `rpk group`/`topic
  describe` mechanics confirmed; a real sustained-throughput consumer-lag chart is a Phase 4 item.
- **Feast contract test (schema-change-fails-CI) not built yet** — deferred to Phase 6 alongside
  the rest of the CI/testcontainers work.
- **Operational finding, not a scope gap:** PyFlink's Python driver spawns a Java child process
  that `timeout`/SIGTERM does not reliably kill — an early smoke-test run sat orphaned for ~1
  hour before being found via `ps aux` and killed manually. Worth checking for stray
  `PythonGatewayServer` processes on this VM before assuming a prior run's resources are freed.
