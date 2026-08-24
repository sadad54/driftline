"""Ground-truth inspection of the IEEE-CIS raw files before any pipeline code is written."""
import pandas as pd

RAW = "data/raw/ieee-cis"

txn = pd.read_csv(f"{RAW}/train_transaction.csv")
ident = pd.read_csv(f"{RAW}/train_identity.csv")

print("=== train_transaction.csv ===")
print("shape:", txn.shape)
print("TransactionDT range:", txn["TransactionDT"].min(), "-", txn["TransactionDT"].max())
print("  in days:", (txn["TransactionDT"].max() - txn["TransactionDT"].min()) / 86400)
print("isFraud rate:", txn["isFraud"].mean())
print("isFraud counts:\n", txn["isFraud"].value_counts())
print("TransactionID range:", txn["TransactionID"].min(), "-", txn["TransactionID"].max())
print("dup TransactionID:", txn["TransactionID"].duplicated().sum())
print("card1-6 dtypes:\n", txn[["card1", "card2", "card3", "card4", "card5", "card6"]].dtypes)
print("addr1/addr2 null rate:", txn["addr1"].isna().mean(), txn["addr2"].isna().mean())
print("P_emaildomain null rate:", txn["P_emaildomain"].isna().mean())
print("R_emaildomain null rate:", txn["R_emaildomain"].isna().mean())
print("DeviceType/DeviceInfo cols exist in txn?", "DeviceType" in txn.columns, "DeviceInfo" in txn.columns)
print("overall null rate (mean across all cols):", txn.isna().mean().mean())
top_null = txn.isna().mean().sort_values(ascending=False).head(10)
print("top-10 nullest columns:\n", top_null)

print("\n=== train_identity.csv ===")
print("shape:", ident.shape)
print("dtypes sample:\n", ident.dtypes.head(15))
print("TransactionID overlap with txn:", ident["TransactionID"].isin(txn["TransactionID"]).mean())
print("fraction of txn rows that have identity info:", txn["TransactionID"].isin(ident["TransactionID"]).mean())
if "DeviceType" in ident.columns:
    print("DeviceType values:\n", ident["DeviceType"].value_counts(dropna=False))
if "DeviceInfo" in ident.columns:
    print("DeviceInfo null rate:", ident["DeviceInfo"].isna().mean())

print("\n=== identity graph entity candidates (in txn) ===")
for col in ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]:
    print(f"{col}: nunique={txn[col].nunique()}, null_rate={txn[col].isna().mean():.4f}")
