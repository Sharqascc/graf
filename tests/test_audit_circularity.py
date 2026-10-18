"""Structural guard for the item-1 circularity audit."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_AUDIT = _REPO / "docs" / "audit" / "feature_label_circularity.md"


def test_audit_document_exists():
    assert _AUDIT.exists()


def test_audit_states_the_verdict():
    text = _AUDIT.read_text()
    assert "TTC-derived information IS present" in text
    assert "near-circular" in text


def test_audit_names_the_nine_edge_columns():
    text = _AUDIT.read_text()
    for col in (
        "edge_attr_mean",
        "edge_attr_std",
        "edge_attr_min",
        "edge_attr_max",
        "edge_row_l2_mean",
        "edge_row_l2_std",
        "edge_row_l2_min",
        "edge_row_l2_max",
        "edge_attr_nonzero_frac",
    ):
        assert col in text, f"audit must name {col}"


def test_audit_names_the_edge_indices_of_ttc():
    text = _AUDIT.read_text()
    for idx in ("2  distance", "5  relative_speed", "6  closing_speed", "11 ttc"):
        assert idx in text, f"audit must cite edge column {idx}"


def test_audit_prescribes_the_clean_path():
    text = _AUDIT.read_text()
    assert "Remove TTC-derived edge features" in text
    assert "forecasting" in text.lower()


def test_feature_names_have_no_literal_ttc_column():
    src = (_REPO / "src" / "graf" / "models" / "baselines.py").read_text()
    start = src.find("_FEATURE_NAMES")
    assert start >= 0
    body = src[start : src.find("]", start)]
    assert chr(34) + "ttc" + chr(34) not in body, (
        "a feature named ttc was added; update the circularity audit"
    )
