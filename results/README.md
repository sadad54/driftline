# Phase 1 baseline results

Run: `scripts/run_baseline.py`, full output in `baseline_metrics.json`. Time-ordered holdout
(last 20% of 590,540 rows by `TransactionDT`) — 472,432 train / 118,108 test, ~3.5% fraud rate
in both (no meaningful class-balance drift yet this early in the 6-month span, as expected).

## Headline baseline number
**XGBoost alone: PR-AUC 0.4751, ROC-AUC 0.8892, recall@1%FPR 37.6%, precision@top-1181 = 81.6%.**
This is the number every later phase (GraphSAGE ensemble ablation, drift-triggered retrain decay
curve) reports lift against.

## Finding: the naive rank-average ensemble is *worse* than XGBoost alone
| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision@k |
|---|---|---|---|---|
| XGBoost | **0.4751** | **0.8892** | **0.3760** | **0.8163** |
| IsolationForest | 0.1193 | 0.7673 | 0.0000 | 0.0000 |
| Rank-average ensemble | 0.4269 | 0.8814 | 0.3501 | 0.7646 |

IsolationForest's top-1181 most-anomalous test transactions contain **zero** actual fraud cases
(`precision_at_k = 0.0`), and its recall at a 1% false-positive threshold is also zero — its
anomaly score is essentially uncorrelated with `isFraud` in this dataset, at these default
hyperparameters (`n_estimators=200`, `contamination="auto"`, ordinal-coded categoricals,
median-imputed numerics). Averaging its rank into the ensemble therefore drags every metric down
relative to XGBoost alone, rather than adding complementary signal.

**Decision:** don't carry this untuned IsolationForest baseline forward into the Phase 3 ensemble
as-is. Two honest options for later: (a) tune IsolationForest specifically (contamination rate,
feature subset, or restrict it to a manifold where anomaly detection plausibly adds value, e.g.
new-card/no-history transactions where XGBoost has less to learn from), or (b) drop it and let
GraphSAGE be the second ensemble member instead, reporting this negative result as the reason.
Either way, this is exactly the kind of "quantify before hand-waving" finding worth keeping for
the interview story about honest ablations — a candidate who just wrote "XGBoost + IsolationForest
ensemble, PR-AUC 0.43" without noticing this would be reporting a worse number without knowing why.

## The "metrics lied to you" artifact (`random_vs_time_split.py`)
Same XGBoost config, same 590,540-row dataset — only the split changes.

| Split | PR-AUC | ROC-AUC | Recall@1%FPR |
|---|---|---|---|
| Time-ordered (honest) | 0.4751 | 0.8892 | 0.3760 |
| Random 80/20, stratified | **0.6974** | **0.9442** | **0.6175** |
| Inflation | **+0.2222** | +0.0550 | **+0.2415** |

A random split lets the same card/address/email-domain entity appear in both train and test —
IEEE-CIS transactions are not i.i.d. across time, entities recur within short windows. The model
partially memorizes entity identity rather than learning genuinely predictive fraud signal, so
the random-split number is badly inflated relative to the honest forward-looking evaluation
production actually faces. Recall@1%FPR alone swings from a defensible 37.6% to a fictitious
61.7% — the exact anti-pattern the source doc's "how candidates do this badly" section warns
about, now with a measured number instead of an assertion. This is the source doc's week-1
interview story ("tell me about a time your metrics lied to you"), built for real.

## Runtime (this build machine, 7.3GB RAM / 8 vCPU, no GPU)
- Data load + merge + downcast + sort: 181.5s
- XGBoost fit (400 trees, 434 features, 472K rows): 89.7s
- IsolationForest fit (200 trees): 15.6s

Note: the initial naive loader (float64 throughout, no downcast) OOM'd on this machine — the
merged frame's dense float64 block alone needed ~1.76GB to consolidate. Fixed in
`src/driftline/data.py::_downcast_numeric` (column-wise float64→float32 / int64→int32 downcast
right after each CSV read, before merge). Documented here rather than silently patched, since
"what fits in memory on real hardware" is itself a legitimate engineering constraint worth being
able to talk about.
