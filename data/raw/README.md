# Raw data

Not committed to git (see `.gitignore`). Populated by `scripts/download_data.py`.

Expected contents after download:
- `ieee-cis/train_transaction.csv`, `ieee-cis/train_identity.csv` — primary dataset (590,540 transactions, 434 features, ~3.5% fraud, 6 months, `TransactionDT` seconds-offset timestamp)
- `elliptic/` — Elliptic Bitcoin graph dataset (203,769 nodes, 234,355 edges) — graph-model supplement
- `paysim/` — PaySim synthetic mobile-money transactions (6.3M rows) — throughput/load-test supplement only, not used for model training
