# Data & leakage rules

## Source
IEEE-CIS Fraud Detection (Kaggle competition), `train_transaction.csv` + `train_identity.csv`,
left-joined on `TransactionID`. See `raw/README.md` for download instructions and file layout.

## Ground-truth facts (verified via `scripts/inspect_ieee_cis.py`, not assumed)
- 590,540 transaction rows, 394 transaction columns + 40 identity columns (434 total, minus the
  shared join key) — matches the source doc's headline numbers.
- Fraud rate: 3.499% (20,663 / 590,540).
- `TransactionDT` is a **relative** offset in seconds (starts at 86,400 = "day 1"), not a real
  calendar timestamp. It spans ~182 days (~6 months). There is no wall-clock date anywhere in
  this dataset — "month" bucketing (`add_month_bucket`) means "1/6th of the observed span", not
  a calendar month.
- Only 24.4% of transactions have a matching row in `train_identity.csv`. The join is **left**,
  not inner — an inner join would silently drop 75.6% of rows, which is itself a bug class the
  anti-pattern section of the source doc would flag.
- `test_transaction.csv` / `test_identity.csv` (Kaggle's official test split) have **no
  `isFraud` labels** — they're the blind leaderboard set. They are never used for our own
  evaluation. All train/holdout splits are carved out of `train_transaction.csv` alone.
- Overall null rate across all columns: ~41%. XGBoost's native NaN handling (`tree_method=hist`)
  is used instead of imputation for the supervised path; IsolationForest (which cannot consume
  NaN) gets a separately median-imputed, ordinal-encoded numeric matrix — this is a real
  difference in preprocessing between the two models, not an oversight.

## Leakage rules (enforced by `tests/test_data.py`, not just documented)
1. **No random train/test split, ever.** All splits go through `time_ordered_split()`, which
   sorts by `TransactionDT` and takes the chronologically last `test_frac` as holdout.
   `test_time_ordered_split_no_leakage` in the test suite asserts
   `train.TransactionDT.max() <= test.TransactionDT.min()` and fails CI if violated.
2. **`TransactionID` and `TransactionDT` are never model features** — see
   `FEATURE_COLS_EXCLUDE` in `src/driftline/data.py`. `TransactionID` is a monotonically
   increasing surrogate key correlated with time; including it would leak the same signal a
   random split would.
3. **The month bucket (`month` column) is derived, not a feature** — it exists for the Phase 5
   drift-decay analysis (PR-AUC by month) and reporting, and is explicitly excluded from
   `feature_columns()`.
4. Any future feature-engineering step (velocity windows, entity-graph features, Feast
   materialization) must be computed **only from rows with `TransactionDT` strictly less than**
   the row being featurized — this becomes its own leakage test once Phase 2/3 code lands.
