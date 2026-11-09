"""Tests for scripts/sci_density_analysis.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.sci_density_analysis import (
    blocked_cv_auc,
    estimate_pixels_per_meter,
    frame_scan_cache,
    label_rates_from_cache,
    label_variants,
    load_tracks,
    permute_tracks,
    scan_frame,
)

REPO = Path(__file__).resolve().parents[1]


def _tracks_jsonl(path, rows):
    import json

    with Path(path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(fid, tid, cls, x1, y1, x2, y2):
    return {
        "frame_idx": fid,
        "track_id": tid,
        "class_name": cls,
        "confidence": 0.9,
        "frame_h": 1080,
        "frame_w": 1920,
        "bbox_xyxy": [x1, y1, x2, y2],
    }


def _tiny_df():
    rows = [
        {
            "frame_idx": fid,
            "track_id": 1,
            "x_m": 0.0,
            "y_m": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "class_name": "car",
            "speed": 0.0,
        }
        for fid in range(10)
    ] + [
        {
            "frame_idx": fid,
            "track_id": 2,
            "x_m": 1.5 - 0.1 * fid,
            "y_m": 0.0,
            "vx": -3.0,
            "vy": 0.0,
            "class_name": "car",
            "speed": 3.0,
        }
        for fid in range(10)
    ]
    return pd.DataFrame(rows)


def test_estimate_pixels_per_meter_recovers_scale(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [
        _row(fid, tid, "car", 100, 100, 300, 550)
        for fid in range(5)
        for tid in range(20)
    ]
    _tracks_jsonl(p, rows)
    df = load_tracks(p)
    ppm = estimate_pixels_per_meter(df)
    assert 80 < ppm < 120


def test_scan_frame_returns_critical_for_closing_pair():
    df = pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "track_id": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "class_name": "car",
            },
            {
                "frame_idx": 0,
                "track_id": 2,
                "x_m": 1.0,
                "y_m": 0.0,
                "vx": -1.0,
                "vy": 0.0,
                "class_name": "car",
            },
        ]
    )
    crit, best_ttc, n_pairs = scan_frame(df, 1.5, 3.0, 0.5, 1.5)
    assert len(crit) == 1
    assert n_pairs == 1
    assert best_ttc == pytest.approx(1.0)


def test_scan_frame_rejects_far_pairs():
    df = pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "track_id": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "class_name": "car",
            },
            {
                "frame_idx": 0,
                "track_id": 2,
                "x_m": 10.0,
                "y_m": 0.0,
                "vx": -1.0,
                "vy": 0.0,
                "class_name": "car",
            },
        ]
    )
    crit, _, n_pairs = scan_frame(df, 1.5, 3.0, 0.5, 1.5)
    assert n_pairs == 0
    assert len(crit) == 0


def test_label_variants_degenerate_when_nothing_critical():
    pf = {i: ([], float("inf"), 1) for i in range(100)}
    labels = label_variants(pf)
    assert labels["any_1.5"] == 0
    assert labels["sustained_r5"] == 0
    assert labels["min_ttc_0.5"] == 0


def test_label_variants_fires_on_persistent_collision():
    pf = {}
    for i in range(100):
        if 50 <= i <= 54:
            pf[i] = ([(1, 2)], 0.2, 1)
        else:
            pf[i] = ([], float("inf"), 1)
    labels = label_variants(pf)
    assert labels["any_1.5"] == 1
    assert labels["sustained_r5"] == 1
    assert labels["sustained_r30"] == 0
    assert labels["min_ttc_0.5"] == 1
    assert labels["persistent_r3"] == 1


def test_blocked_cv_auc_detects_signal():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 4)
    y = (X[:, 0] + 0.2 * rng.randn(n) > 0).astype(int)
    auc = blocked_cv_auc(X, y, purge=5, seed=42)
    assert auc is not None and auc > 0.65


def test_blocked_cv_auc_returns_none_for_small_input():
    X = np.random.randn(20, 3)
    y = np.array([0, 1] * 10)
    assert blocked_cv_auc(X, y) is None


def test_blocked_cv_auc_returns_none_for_degenerate_labels():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    assert blocked_cv_auc(X, np.zeros(200, dtype=int)) is None
    assert blocked_cv_auc(X, np.ones(200, dtype=int)) is None


def test_cli_runs_on_synthetic_tracks(tmp_path):
    rows = []
    for fid in range(60):
        t = fid / 30.0
        rows.append(_row(fid, 1, "car", 50, 550, 150, 650))
        x2 = max(4 - 4 * t, 0.5)
        rows.append(
            _row(fid, 2, "car", int(x2 * 100 - 50), -50, int(x2 * 100 + 50), 50)
        )
        x3 = min(-4 + 4 * t, -0.5)
        rows.append(
            _row(fid, 3, "car", int(x3 * 100 - 50), -50, int(x3 * 100 + 50), 50)
        )
    p = tmp_path / "tracks.jsonl"
    _tracks_jsonl(p, rows)
    out = tmp_path / "out"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "sci_density_analysis.py"),
        "--tracks",
        str(p),
        "--pixels_per_meter",
        "100",
        "--output_dir",
        str(out),
        "--win_frames",
        "30",
        "--stride_frames",
        "5",
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, f"stdout: {r.stdout[-800:]}\nstderr: {r.stderr[-800:]}"
    csv = out / "window_features.csv"
    assert csv.exists()
    W = pd.read_csv(csv)
    assert len(W) == 6
    for col in (
        "any_1.5",
        "sustained_r5",
        "sustained_r30",
        "min_ttc_0.5",
        "min_ttc_1.0",
        "pair_frac_0.05",
        "pair_frac_0.10",
        "persistent_r3",
    ):
        assert col in W.columns
    assert (W["any_1.5"] == 1).all()
    assert (W["sustained_r30"] == 0).all()


# ------------------------------------------------------------------
# Permutation test
# ------------------------------------------------------------------


def test_permute_tracks_preserves_durations():
    df = _tiny_df()
    before = df.groupby("track_id").size().to_dict()
    rng = np.random.default_rng(42)
    permuted = permute_tracks(df, rng)
    after = permuted.groupby("track_id").size().to_dict()
    assert before == after


def test_permute_tracks_keeps_frames_in_clip():
    df = _tiny_df()
    rng = np.random.default_rng(42)
    permuted = permute_tracks(df, rng)
    assert permuted["frame_idx"].min() >= df["frame_idx"].min()
    assert permuted["frame_idx"].max() <= df["frame_idx"].max()


def test_permute_tracks_deterministic_for_seed():
    df = _tiny_df()
    a = permute_tracks(df, np.random.default_rng(42))
    b = permute_tracks(df, np.random.default_rng(42))
    a_sorted = a.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    b_sorted = b.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    assert a_sorted.equals(b_sorted)


def test_permute_tracks_changes_frame_positions():
    """Circular shift preserves the SET of frame indices, so compare
    unsorted sequences (a nonzero offset rotates the sequence)."""
    df = _tiny_df()
    rng = np.random.default_rng(42)
    permuted = permute_tracks(df, rng)
    orig_by_tid = (
        df.groupby("track_id")["frame_idx"].apply(lambda s: tuple(s.tolist())).to_dict()
    )
    perm_by_tid = (
        permuted.groupby("track_id")["frame_idx"]
        .apply(lambda s: tuple(s.tolist()))
        .to_dict()
    )
    n_shifted = sum(1 for tid in orig_by_tid if orig_by_tid[tid] != perm_by_tid[tid])
    assert n_shifted >= 1


def test_permutation_on_sparse_scene_stays_zero():
    """20 stationary, well-separated actors: no collisions real or null."""
    fmin, fmax = 0, 99
    rows = [
        {
            "frame_idx": fid,
            "track_id": tid,
            "x_m": 50.0 + tid * 10.0,
            "y_m": 50.0,
            "vx": 0.0,
            "vy": 0.0,
            "class_name": "car",
            "speed": 0.0,
        }
        for tid in range(20)
        for fid in range(fmin, fmax + 1)
    ]
    df = pd.DataFrame(rows)
    real_cache = frame_scan_cache(df, fmin, fmax)
    real_rates = label_rates_from_cache(real_cache, fmin, fmax, 30, 10)
    assert real_rates["any_1.5"] == 0.0
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        permuted = permute_tracks(df, rng)
        cache = frame_scan_cache(permuted, fmin, fmax)
        rates = label_rates_from_cache(cache, fmin, fmax, 30, 10)
        assert rates["any_1.5"] == 0.0


def test_permutation_rates_bounded():
    df = _tiny_df()
    rng = np.random.default_rng(42)
    permuted = permute_tracks(df, rng)
    fmin = int(df["frame_idx"].min())
    fmax = int(df["frame_idx"].max())
    cache = frame_scan_cache(permuted, fmin, fmax)
    rates = label_rates_from_cache(cache, fmin, fmax, 10, 5)
    for rate in rates.values():
        assert 0.0 <= rate <= 1.0
