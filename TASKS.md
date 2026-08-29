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

- [ ] Stand up Redpanda (Kafka-API compatible) via docker-compose; create topic `transactions.raw`
      with sensible partition count
- [ ] Build `replay_producer.py`: emits IEEE-CIS events in strict `TransactionDT` order at a
      configurable events/sec rate; supports pause/resume and rate override via CLI flag
- [ ] Verify producer correctness: consumed event order matches source order; no dropped/duplicated
      events at steady rate
- [ ] Implement consumer group(s) and confirm consumer-lag is observable (via `rpk` or exposed
      metric) — this is a named differentiator ("actual consumer lag metric") vs. fake real-time
- [ ] Implement windowed aggregation job in **PyFlink**: card/device/email velocity over 1h / 24h /
      7d tumbling/sliding windows
- [ ] Keep a Spark Structured Streaming variant behind a feature flag/alt entrypoint (claim both
      truthfully per the doc's guidance — build it for real or explicitly mark it as not built in
      Known Gaps, don't claim it either way falsely)
- [ ] Wire windowed aggregate output to two sinks: Redis (Feast online store) and Parquet (Feast
      offline store)
- [ ] Stand up Feast: define entities (card, device, email, addr) and feature views for the
      windowed aggregates; register online (Redis) + offline (Parquet) stores
- [ ] Write and pass **training-serving skew test**: sample N replayed events, compare offline vs.
      online feature values, assert max absolute difference; document the actual skew bug found and
      fixed (the doc calls this out explicitly — find and record a real one, don't fabricate)
- [ ] Contract test on Feast feature definitions: a schema change to a feature view must fail CI
- [ ] Measure and record **feature freshness lag** (event time → online-store availability), p99,
      on the EC2 hardware

---

## Phase 3 — Graph Model

- [ ] Construct entity graph from IEEE-CIS identity columns: card1-6, addr1-2, device info, email
      domain as nodes/shared-identity edges
- [ ] Persist graph construction as a reproducible script (not notebook-only), with node/edge counts
      logged
- [ ] Implement **temporal edge masking** so training at time T never sees edges formed after T (no
      future-edge leakage) — write a test asserting this
- [ ] Build PyTorch Geometric GraphSAGE model on the entity graph; train with same time-ordered
      split as the tabular baseline
- [ ] Optional/stretch: reproduce a published GraphSAGE baseline result on Elliptic Bitcoin Dataset
      as a sanity check on the graph pipeline's correctness (independent of IEEE-CIS)
- [ ] Build rank-average ensemble combining XGBoost + GraphSAGE (+ IsolationForest residual signal)
- [ ] Produce **ablation table**: XGBoost alone vs. XGBoost+GraphSAGE vs. full ensemble, PR-AUC and
      recall@1%FPR for each, with a named baseline row
- [ ] Compute **graph lift specifically on the "new card, no history" slice** — isolate this
      segment and report PR-AUC delta there vs. overall; this is the doc's called-out
      differentiator, don't skip it
- [ ] Record: does GraphSAGE help everywhere or only on specific slices? Write the honest
      "where it didn't help" note for the interview story

---

## Phase 4 — Serving

- [ ] Export XGBoost + GraphSAGE (+ IsolationForest) to ONNX; write an ONNX-vs-native parity test
      (predictions match within tolerance)
- [ ] Build FastAPI scoring service: consumes `transactions.raw` (or scores on request), fetches
      Feast online features, runs ensemble via ONNX Runtime, publishes to `transactions.scored`
- [ ] Containerize scoring service; run it in docker-compose locally, then in k3d
- [ ] Locust load test against the scoring service; sustained events/sec, p50/p95/p99 end-to-end
      latency, on stated EC2 hardware — record exact numbers, not estimates
- [ ] Implement alert-queue / precision@k simulation: fixed daily analyst review budget (k),
      compute precision@k on the replay — this answers the "10,000 flags, 200 analysts" interview
      question for real
- [ ] End-to-end smoke test: producer → Redpanda → Flink → Feast → scorer → `transactions.scored`,
      verified on a small replay batch

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

- 
