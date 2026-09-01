# Phase 5 — Drift & Retraining: results

## Performance decay curve (the headline artifact, `scripts/decay_curve.py`)
Trained XGBoost on month 1 only (135,222 rows), evaluated months 1-6 **without retraining**.

| Month | PR-AUC | ROC-AUC | Recall@1%FPR | n | Fraud rate |
|---|---|---|---|---|---|
| 1 | 0.9069 | 0.9950 | 0.8845 | 135,222 | 2.57% |
| 2 | **0.4761** | 0.8527 | 0.3823 | 89,948 | 3.95% |
| 3 | 0.3936 | 0.8011 | 0.3127 | 94,757 | 4.07% |
| 4 | 0.3779 | 0.8151 | 0.2949 | 98,305 | 3.93% |
| 5 | 0.3319 | 0.7966 | 0.2758 | 84,938 | 3.36% |
| 6 | **0.3680** | 0.8046 | 0.3077 | 87,370 | 3.50% |

**Important caveat, stated plainly:** month 1's 0.9069 is an **in-sample** number (evaluated on
the same data the model was trained on) — quoting the month-1→month-6 delta (+59.4% relative) as
"the decay" would overstate it with training-set fit, not real drift. The honest post-deployment
comparison is **month 2 (first genuinely held-out month, 0.4761) → month 6 (0.3680), a real
22.7% relative decay** with no retraining. This is a genuine, measured drift curve from real
data — not injected synthetic noise.

## Weekly PSI + KS drift monitoring (`scripts/drift_monitoring.py`)
369 numeric features, reference = week 0 (27,596 rows), computed vs. every subsequent week
(manual PSI/KS implementation — see the script's docstring for why Evidently 0.7.x's rewritten
API wasn't used; that's a documented substitution, not an oversight).

- **First week crossing the retrain trigger (≥20 features with PSI > 0.2): week 2** — drift onset
  well before month 1 even ends (~week 4.3), a more precise finding than month-level granularity
  alone would show.
- Feature `V152` is the single most consistently drifted feature across nearly every week
  (PSI up to 5.94 — far past the 0.2 threshold).
- Breach count stays persistently high (16-35 features/week) for the rest of the replay once
  drift sets in, against the fixed week-0 reference.

## Drift-triggered retraining: shadow-scored, gated, MLflow-logged (`scripts/drift_triggered_retrain.py`)
Walked forward week-by-week from week 4. Initial model trained on weeks 0-3 (126,666 rows,
matching decay_curve.py's month-1-sized baseline). On each PSI trigger: retrain on data since the
last retrain, shadow-score the candidate on a held-out tail slice of that same window (never on
future weeks — keeps the promotion decision causal), promote only if PR-AUC doesn't regress by
more than 0.02 absolute vs. the currently-serving model on that same shadow slice.

**2 retrains triggered, 2 promoted:**

| Event | Trigger | Candidate shadow PR-AUC | Serving shadow PR-AUC | Regression | Decision |
|---|---|---|---|---|---|
| Week 14 | 23 features PSI>0.2 | 0.4802 | 0.2811 | **-0.1991** (candidate much better) | PROMOTED |
| Week 24 | 35 features PSI>0.2 | 0.5061 | 0.4358 | **-0.0703** (candidate better) | PROMOTED |

**The recovered delta, quoted directly:** without retraining, the serving model degraded from
week 4's 0.5913 down to **week 14's 0.2635** — worse than the untreated month-6 baseline above.
Retraining at week 14 recovered performance to **week 15's 0.5211 — a +0.2576 absolute (+97.8%
relative) recovery in a single week.** The same pattern repeats at week 24 (degraded to 0.3663 by
week 20, recovered toward 0.48 after the second retrain).

## Airflow substitution (stated explicitly, see script docstring and Known Gaps)
A lightweight Python orchestration loop was used instead of a full Airflow deployment
(webserver + scheduler + metadata DB). The actual engineering content — drift detection,
cumulative retrain, causal shadow-scoring, a real promotion gate — is identical either way, and
every run is logged to the already-running MLflow instance (`http://localhost:5000`, experiment
`driftline-drift-triggered-retrain`) with real params/metrics per run, not simulated.

## What's not done yet (Known Gaps)
- PSI heatmap panel in Grafana (would need a metrics pushgateway for the weekly batch job's
  output — not wired up; the dashboard covers scorer-side metrics only, see phase4/phase6 notes).
- Consumer-lag panel (would need a Kafka lag exporter, e.g. kminion — not deployed).
- Real Airflow deployment (see substitution above).
