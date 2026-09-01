# Driftline — interview-story answers

Each answer below is backed by a specific, runnable artifact in this repo — not a rehearsed line.
Point to the file/number if asked to go deeper.

---

### "How do you know your model is still working in production?"
I don't wait to find out from a downstream metric — I monitor input drift directly. Every week,
I compute PSI and KS for all 369 numeric features against a fixed reference window
(`scripts/drift_monitoring.py`). On this data, the retrain trigger (20+ features crossing
PSI > 0.2) fires at **week 2** — well before even the first month of "production" ends. I chose
0.2 as the PSI threshold because it's the standard industry rule-of-thumb boundary between
"moderate" and "significant" distribution shift; I can defend a different threshold if asked, but
this one is a deliberate choice, not a default I left alone.

### "Fraudsters adapt. How do you handle that?"
I have a *measured* adversarial-style drift, not a hypothetical: training on month 1 only and
evaluating forward with no retraining, PR-AUC decays from 0.4761 (first held-out month) to 0.3680
by month 6 — a real 22.7% relative decline (`results/decay_curve.json`). My response isn't
manual — `scripts/drift_triggered_retrain.py` walks forward automatically, and when it actually
fires (twice, in the real run), it recovers a mid-stream collapse from PR-AUC 0.2635 to 0.5211 in
a single week. That recovery is the actual answer to "how do you handle it": automatically,
with a gate that stops a bad retrain from shipping (see the promotion-gate story below).

### "Why a graph model here?"
IEEE-CIS transactions share identity through card/address/email fields — a real shared-identity
graph, not a contrived one. I built a heterogeneous entity graph (605K nodes: 590K transactions +
15K identity-value anchors) and trained GraphSAGE with a *tested* leakage boundary (
`tests/test_graph.py` — the training graph structurally cannot reference a test-set row, not just
by convention). Honest result: **the naive ensemble doesn't beat XGBoost alone** (PR-AUC 0.4687 vs
0.4751). I know exactly why — equal-weight rank-averaging punishes a weaker signal instead of
learning how much to trust it, and 5 CPU-only epochs genuinely undertrains a GNN relative to a
tuned 400-tree XGBoost. The fix (a small logistic-regression stacker on a held-out validation
slice) is specified, not vague, and logged as the next step. Separately, a real independent
finding survived regardless of the ensemble result: new-card transactions have a **34% higher**
fraud rate (4.62% vs. 3.44% overall) — exactly where you'd expect identity-graph signal to matter
most, which is itself useful even without the GNN lift.

### "10,000 flags a day, 200 analysts. What now?"
That's a ~2% review budget. On this data (`scripts/analyst_budget_simulation.py`, real per-day
numbers across 42 test days, ~2,812 transactions/day mean volume): at a 2% budget (~56
reviews/day), mean precision@k is **59.6%** — so a bit under 6 in 10 flagged transactions an
analyst reviews are actually fraud. At a tighter 0.5% budget, precision climbs to 87.7%. I can
show the real day-to-day variance too (33%-100% at the 0.5% budget) — the mean alone would
understate how much this swings with actual daily volume and fraud clustering.

### "What's training-serving skew, and have you ever hit it?"
Yes — a real one, not a definition I memorized. My Flink job writes features to two places per
event: a real-time Redis `hset` and a 200-row-buffered Parquet flush (a deliberate
durability-vs-latency tradeoff — buffering risks losing at most one buffer's worth of rows on an
ungraceful kill, instead of the entire file becoming invalid, which is a *different* real bug I
found and fixed along the way). Comparing the two paths directly
(`scripts/drift_monitoring.py`'s sibling, `feature_store/skew_test_realtime.py`): **240/2,772
(8.66%) of sampled (card1, window) pairs disagree**, because online is briefly ahead of offline —
the reverse of the usual assumption that online lags offline. I found this by testing the right
pair (Feast-materialize-vs-source is a straight copy and can never diverge — I tested that too,
and correctly got 0 mismatches, which taught me it was the wrong comparison to make the point).

### "Tell me about a time your metrics lied to you."
Same XGBoost, same 590K-row dataset, only the split strategy changed
(`results/random_vs_time_split.json`): a random 80/20 split scores PR-AUC 0.6974; the honest
time-ordered split scores 0.4751. A **+0.2222 absolute inflation from nothing but letting rows
from the same card/device/email appear in both train and test** — the model partially memorizes
entity identity instead of learning genuinely predictive fraud patterns. Recall@1%FPR swings even
harder: 61.7% (random) vs. 37.6% (honest). I quote both numbers together deliberately, because
the gap itself — not either number alone — is the actual finding.

---

## Two more I'd bring up unprompted if the interview goes well

**"What broke when you actually load-tested this?"** — I drove 100 concurrent users at the
k3d-deployed scorer and got a real HPA scale event (2→4 replicas, confirmed via `kubectl` events)
*and* a real production risk I hadn't anticipated: nearly every pod failed its liveness probe
simultaneously because a single uvicorn worker was too busy scoring requests to answer a trivial
health check, and kubelet killed one. That's the same single-worker-queueing finding from the
docker-compose load test (p50 latency 21ms single-request vs. 240ms under 50-user load) showing
up one layer higher, with a concrete consequence (an unplanned restart) instead of just slower
responses. The fix (`uvicorn --workers N`) is identified, not yet re-benchmarked — I'd rather say
that plainly than imply it's solved.

**"Walk me through a bug you found and fixed."** — Several real candidates, but the cleanest:
`ParquetWriter` with a single incrementally-appended file produces a file with **no footer** —
permanently invalid — if the process is killed via SIGTERM instead of a graceful shutdown. That's
exactly how a streaming job actually gets stopped in the real world, not an edge case. I found it
by testing the failure path, not just the happy path. Fixed by buffering 200 rows and flushing
each batch as its own complete, independently-valid part-file — bounding data loss to one
unflushed buffer instead of losing the entire file's contents.
