"""Diagnostic tests for the rel_heading_* edge features."""

from __future__ import annotations

import math

import pytest

from graf.graph.edges import build_edge_feature


def test_heading_available_when_input_has_it():
    src = {
        "x": 0.0,
        "y": 0.0,
        "vx": 1.0,
        "vy": 0.0,
        "heading_rad": 0.0,
    }
    dst = {
        "x": 2.0,
        "y": 0.0,
        "vx": -1.0,
        "vy": 0.0,
        "heading_rad": 0.3,
    }
    feat = build_edge_feature(src, dst)
    assert feat["rel_heading_sin"] == pytest.approx(math.sin(0.3), abs=1e-12)
    assert feat["rel_heading_cos"] == pytest.approx(math.cos(0.3), abs=1e-12)


def test_heading_zero_when_input_lacks_it():
    src = {"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0}
    dst = {"x": 2.0, "y": 0.0, "vx": -1.0, "vy": 0.0}
    feat = build_edge_feature(src, dst)
    assert feat["rel_heading_sin"] == 0.0
    assert feat["rel_heading_cos"] == 1.0


def test_vntraffic_edge_tensors_have_zero_rel_heading_sin():
    """The committed fixture has no heading_rad, so its edge tensors
    have rel_heading_sin identically zero."""
    from pathlib import Path

    import torch

    fixture_dir = (
        Path(__file__).resolve().parents[1] / "data" / "fixtures" / "demo" / "graphs"
    )
    graph_files = sorted(fixture_dir.glob("*.pt"))
    assert graph_files, f"no fixture graphs at {fixture_dir}"
    g = torch.load(graph_files[0], weights_only=False)
    rel_heading_sin = g.edge_attr[:, 9]
    assert torch.all(rel_heading_sin == 0.0).item()


def test_reverse_edge_feature_negates_sin_not_cos():
    from graf.graph.edges import build_edge_feature, reverse_edge_feature

    src = {"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0, "heading_rad": 0.0}
    dst = {"x": 2.0, "y": 0.0, "vx": -1.0, "vy": 0.0, "heading_rad": 0.4}
    forward = build_edge_feature(src, dst)
    reverse = reverse_edge_feature(forward)
    assert reverse["rel_heading_sin"] == pytest.approx(-forward["rel_heading_sin"])
    assert reverse["rel_heading_cos"] == pytest.approx(forward["rel_heading_cos"])
