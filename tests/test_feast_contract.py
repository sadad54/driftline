"""Feast schema contract test: a breaking change to a feature view's schema must fail CI, not
silently ship. Doesn't require a live Feast/Redis connection (pytest runs in GitHub Actions where
none is available) -- validates the FeatureView objects' declared schema directly against the
contract this repo depends on elsewhere (skew_test.py, the FastAPI scorer's Feast lookup).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature_store"))

from definitions import card_velocity_1h_fv, card_velocity_24h_fv, card_velocity_7d_fv  # noqa: E402

EXPECTED_FEATURE_VIEWS = {
    "card_velocity_1h": {"txn_count", "amt_sum"},
    "card_velocity_24h": {"txn_count", "amt_sum"},
    "card_velocity_7d": {"txn_count", "amt_sum"},
}


def _schema_fields(fv) -> set:
    return {f.name for f in fv.schema}


def test_feature_view_names_match_contract():
    actual_names = {fv.name for fv in (card_velocity_1h_fv, card_velocity_24h_fv, card_velocity_7d_fv)}
    assert actual_names == set(EXPECTED_FEATURE_VIEWS.keys())


def test_feature_view_schemas_match_contract():
    """This is the test that actually catches a breaking schema change: if someone renames or
    removes txn_count/amt_sum from a feature view (e.g. while extending Phase 2 to
    DeviceInfo/email), every downstream consumer (skew_test.py, serving/app.py's Feast lookup)
    would silently start returning None/erroring at runtime instead of failing in CI."""
    for fv in (card_velocity_1h_fv, card_velocity_24h_fv, card_velocity_7d_fv):
        expected = EXPECTED_FEATURE_VIEWS[fv.name]
        actual = _schema_fields(fv)
        assert actual == expected, f"{fv.name}: schema drifted from contract ({actual} != {expected})"


def test_all_feature_views_have_card1_entity():
    for fv in (card_velocity_1h_fv, card_velocity_24h_fv, card_velocity_7d_fv):
        # fv.entities holds whatever was passed to the FeatureView constructor -- Entity objects
        # or plain names, depending on Feast version -- so check both forms.
        entity_repr = {getattr(e, "name", e) for e in fv.entities}
        assert "card1" in entity_repr, f"{fv.name}: expected card1 entity, got {entity_repr}"
