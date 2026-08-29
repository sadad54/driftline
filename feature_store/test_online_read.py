"""Sanity check: read features back through the Feast SDK, not just via raw Redis."""
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# card1=13413 appeared in the earlier check_parquet.py sample
entity_rows = [{"card1": 13413}, {"card1": 2263}, {"card1": 999999999}]  # last one shouldn't exist

result = store.get_online_features(
    features=[
        "card_velocity_1h:txn_count",
        "card_velocity_1h:amt_sum",
        "card_velocity_24h:txn_count",
        "card_velocity_7d:txn_count",
    ],
    entity_rows=entity_rows,
    full_feature_names=True,
).to_dict()

for i, card in enumerate([r["card1"] for r in entity_rows]):
    print(f"card1={card}: 1h_count={result['card_velocity_1h__txn_count'][i]}, "
          f"1h_amt={result['card_velocity_1h__amt_sum'][i]}, "
          f"24h_count={result['card_velocity_24h__txn_count'][i]}, "
          f"7d_count={result['card_velocity_7d__txn_count'][i]}")
