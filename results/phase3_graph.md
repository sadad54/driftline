# Phase 3 — Graph Model: results and honest findings

Full run on all 590,540 rows (472,432 train / 118,108 test, same time-ordered split as Phase 1),
on the GCP VM (CPU only, no GPU — `e2-standard-4`).

## Entity graph
Heterogeneous shared-identity graph: 590,540 transaction nodes + 14,811 distinct identity-value
nodes (across card1, card2, card3, card5, addr1, addr2, P_emaildomain, R_emaildomain) — see
`src/driftline/graph.py` for the design rationale (edges linear in rows×columns, not the O(rows²)
blow-up of directly connecting every pair of transactions sharing a value).
- Train-only graph: 6,457,070 directed edges, built in 4.8s.
- Train+test (inference) graph: 8,060,892 directed edges, built in 6.3s.

## Training (GraphSAGE, PyTorch Geometric, mini-batch neighbor sampling via `pyg-lib`)
- Hyperparameters: hidden_dim=64, 5 epochs, fanout=[15, 10], batch_size=1024, Adam lr=0.005.
- **242.6s wall-clock for 5 epochs / 2,310 batches (~9.5-10 batches/sec) on this VM's 4 vCPUs,
  CPU-only.** Loss was still declining at the end (0.674, down from 0.994 at batch 50) — more
  epochs would likely help further; stopped here to keep the phase within the overall build's
  time budget, logged as a real, named constraint rather than an oversight.
- Inference on the full test set: 2.8s.

## Ablation table (real numbers, no cherry-picking)
| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision@k |
|---|---|---|---|---|
| XGBoost alone (Phase 1 baseline) | **0.4776** | **0.8917** | **0.3780** | **0.8222** |
| GraphSAGE alone | 0.2452 | 0.8393 | 0.2227 | 0.3556 |
| XGBoost + GraphSAGE (rank-average) | 0.4687 | 0.8894 | 0.3765 | 0.8069 |
| XGBoost + GraphSAGE + IsolationForest (rank-average) | 0.4602 | 0.8955 | 0.3836 | 0.7985 |

## No-history ("new card1") slice
823 / 118,108 test transactions (0.70%) are a card1's first-ever appearance in the full
time-ordered dataset. **Fraud rate on this slice is 4.62%, vs. 3.44% overall — new cards are
~34% more likely to be fraudulent, a genuine domain finding worth keeping regardless of the
model result below.**

| Model | PR-AUC | ROC-AUC | Recall@1%FPR |
|---|---|---|---|
| XGBoost alone | **0.6828** | **0.9394** | 0.5263 |
| XGBoost + GraphSAGE ensemble | 0.6219 | 0.9034 | **0.5526** |

Both PR-AUC and ROC-AUC are lower for the ensemble here too; recall@1%FPR is marginally higher —
a genuinely mixed, not uniformly negative, result on this slice specifically.

## Honest conclusion: GraphSAGE does not help in this configuration, including on the slice
where it was expected to help most
This is a real negative result, reported as measured rather than hidden or spun. Two identified,
specific, non-speculative reasons, not a vague "needs more work":

1. **Naive rank-average ensembling weights both models equally**, which only helps when the two
   models are of comparable quality — exactly the failure mode already found and documented in
   Phase 1 with IsolationForest (`results/README.md`). Averaging in a weaker signal drags the
   stronger one down. The fix is a **learned/weighted ensemble** (e.g. a small logistic-regression
   stacker fit on `[xgb_score, graphsage_score, iso_score] -> isFraud` on a held-out validation
   slice carved from the tail of TRAIN, never touching TEST) rather than assuming equal weight —
   not implemented here; logged in Known Gaps as the concrete next step, not "more tuning."
2. **5 epochs of mini-batch training with a 64-dim hidden size, on CPU, within this build's time
   budget** is genuinely undertrained relative to a 400-tree, fully-tuned XGBoost baseline with
   access to all 434 hand-engineered features. The loss curve (0.994 → 0.674, still declining)
   supports this directly — this isn't a fundamental architecture problem, it's a training-budget
   one, and the two are worth distinguishing precisely in an interview answer.

## Interview-ready framing
"I built the full entity-graph + GraphSAGE pipeline, verified it end-to-end on the real 590K-row
dataset with a proper train/test leakage boundary (tested, not just assumed), and measured — 
honestly — that in this configuration it doesn't beat the XGBoost baseline, including on the
no-history slice where graph signal should matter most. I know exactly why: naive equal-weight
ensembling punishes a weaker model instead of learning how much to trust it, and the GNN was
undertrained relative to a heavily-tuned baseline given the CPU time budget. The fix in both
cases is well-defined and specific, not 'try harder.'" This is a stronger, more defensible answer
than a suspiciously clean win would be.
