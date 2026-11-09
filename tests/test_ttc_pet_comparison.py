"""Tests for scripts/ttc_pet_comparison.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.ttc_pet_comparison import (
    agreement_summary,
    compute_pair_metrics,
    scan_candidate_pairs,
)


def _track(tid, xs, ys, vxs, vys, start_frame=0):
    return [
        {
            "frame_idx": start_frame + i,
            "track_id": tid,
            "x_m": xs[i],
            "y_m": ys[i],
            "vx": vxs[i],
            "vy": vys[i],
            "class_name": "car",
            "speed": float(np.sqrt(vxs[i] ** 2 + vys[i] ** 2)),
        }
        for i in range(len(xs))
    ]


def _df(*tracks):
    rows = []
    for t in tracks:
        rows.extend(t)
    return pd.DataFrame(rows)


def test_scan_returns_pair_when_close():
    a = _track(1, [0.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    b = _track(2, [2.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    pairs = scan_candidate_pairs(_df(a, b), dist_threshold=3.0)
    assert pairs == {(1, 2)}


def test_scan_empty_when_all_far():
    a = _track(1, [0.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    b = _track(2, [10.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    pairs = scan_candidate_pairs(_df(a, b), dist_threshold=3.0)
    assert pairs == set()


def test_scan_canonicalizes_pair_order():
    a = _track(10, [0.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    b = _track(3, [1.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    pairs = scan_candidate_pairs(_df(a, b), dist_threshold=3.0)
    assert pairs == {(3, 10)}


def test_compute_pair_metrics_ttc_critical():
    """Head-on approach, TTC fires below threshold."""
    a = _track(1, [0.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10)
    b = _track(
        2, [2.0 - 0.05 * i for i in range(10)], [0.0] * 10, [-1.5] * 10, [0.0] * 10
    )
    r = compute_pair_metrics(_df(a, b), 1, 2, ttc_threshold=1.5, pet_threshold=5.0)
    assert r is not None
    assert r["ttc_critical"] is True
    assert r["min_ttc"] <= 1.5
    assert r["min_dist"] < 2.0


def test_compute_pair_metrics_already_close_disagreement():
    """Two actors inside collision radius: TTC returns inf, PET returns 0."""
    a = _track(1, [0.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10)
    b = _track(2, [1.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10)
    r = compute_pair_metrics(_df(a, b), 1, 2, ttc_threshold=1.5, pet_threshold=5.0)
    assert r is not None
    assert not np.isfinite(r["min_ttc"])
    assert r["ttc_critical"] is False
    assert r["pet_critical"] is True
    assert r["pet"] <= 5.0


def test_compute_pair_metrics_neither_critical():
    a = _track(1, [0.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10)
    b = _track(2, [10.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10)
    r = compute_pair_metrics(_df(a, b), 1, 2)
    assert r is not None
    assert r["ttc_critical"] is False
    assert r["pet_critical"] is False


def test_compute_pair_metrics_insufficient_data():
    a = _track(1, [0.0], [0.0], [0.0], [0.0])
    b = _track(2, [1.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10)
    r = compute_pair_metrics(_df(a, b), 1, 2)
    assert r is None


def test_agreement_summary_perfect_agreement():
    df = pd.DataFrame(
        [
            {"ttc_critical": True, "pet_critical": True},
            {"ttc_critical": True, "pet_critical": True},
            {"ttc_critical": False, "pet_critical": False},
        ]
    )
    s = agreement_summary(df)
    assert s["n_pairs"] == 3
    assert s["both"] == 2
    assert s["neither"] == 1
    assert s["ttc_only"] == 0
    assert s["pet_only"] == 0
    assert s["jaccard"] == 1.0


def test_agreement_summary_perfect_disagreement():
    df = pd.DataFrame(
        [
            {"ttc_critical": True, "pet_critical": False},
            {"ttc_critical": False, "pet_critical": True},
        ]
    )
    s = agreement_summary(df)
    assert s["both"] == 0
    assert s["ttc_only"] == 1
    assert s["pet_only"] == 1
    assert s["jaccard"] == 0.0


def test_agreement_summary_empty():
    s = agreement_summary(pd.DataFrame())
    assert s["n_pairs"] == 0
    assert s["jaccard"] == 0.0


def test_agreement_summary_partial_jaccard():
    # both=2, ttc_only=1, pet_only=1 -> jaccard = 2 / 4 = 0.5
    df = pd.DataFrame(
        [
            {"ttc_critical": True, "pet_critical": True},
            {"ttc_critical": True, "pet_critical": True},
            {"ttc_critical": True, "pet_critical": False},
            {"ttc_critical": False, "pet_critical": True},
        ]
    )
    s = agreement_summary(df)
    assert abs(s["jaccard"] - 0.5) < 1e-9
