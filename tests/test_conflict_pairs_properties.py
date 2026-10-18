"""Property tests for graf.training.conflict_pairs (item 9).

The three functions tested here — load_tracks, filter_tracks,
add_world_coords — are exercised indirectly by every downstream
result, but coverage of the module was 44% before this file.

Invariants:
  - filter_tracks never increases row count
  - filter_tracks output satisfies its own thresholds
  - filter_tracks is monotone in min_conf
  - add_world_coords preserves (track_id, frame_idx) pairs
  - add_world_coords derives t_sec from frame_idx exactly
  - add_world_coords derives speed_mps from vx, vy exactly
  - add_world_coords gives first row of each track zero velocity
  - add_world_coords under identity H gives x_m=x_center, y_m=y_bottom
  - load_tracks roundtrips JSONL row count
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graf.training.conflict_pairs import (
    add_world_coords,
    filter_tracks,
    load_tracks,
)


# --- strategies --------------------------------------------------

_xy = st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False)
_wh = st.floats(1.0, 200.0, allow_nan=False, allow_infinity=False)
_conf = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
_track_id = st.integers(0, 20)
_frame = st.integers(0, 100)


@st.composite
def bbox(draw):
    x = draw(_xy)
    y = draw(_xy)
    w = draw(_wh)
    h = draw(_wh)
    return [x, y, x + w, y + h]


@st.composite
def track_row(draw):
    return {
        "video_id": "v",
        "frame_idx": draw(_frame),
        "track_id": draw(_track_id),
        "confidence": draw(_conf),
        "bbox_xyxy": draw(bbox()),
        "class_name": "car",
    }


@st.composite
def track_frame(draw, n_max: int = 10):
    return draw(st.lists(track_row(), min_size=0, max_size=n_max))


def _make_df(rows):
    if not rows:
        return pd.DataFrame(
            columns=[
                "video_id",
                "frame_idx",
                "track_id",
                "confidence",
                "bbox_xyxy",
                "class_name",
            ]
        )
    return pd.DataFrame(rows)


SCALE_PPM = 100.0
H_SCALE = np.array(
    [
        [1.0 / SCALE_PPM, 0.0, 0.0],
        [0.0, 1.0 / SCALE_PPM, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


# --- filter_tracks ----------------------------------------------


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=50, deadline=None)
def test_filter_never_increases_rows(rows):
    df = _make_df(rows)
    out = filter_tracks(df, min_conf=0.0, min_len=1)
    assert len(out) <= len(df)


@pytest.mark.hypothesis
@given(rows=track_frame(), min_conf=st.floats(0.0, 1.0))
@settings(max_examples=50, deadline=None)
def test_filter_respects_confidence(rows, min_conf):
    df = _make_df(rows)
    out = filter_tracks(df, min_conf=min_conf, min_len=1)
    assert (out["confidence"] >= min_conf).all()


@pytest.mark.hypothesis
@given(rows=track_frame(), min_len=st.integers(1, 5))
@settings(max_examples=50, deadline=None)
def test_filter_respects_min_len(rows, min_len):
    df = _make_df(rows)
    out = filter_tracks(df, min_conf=0.0, min_len=min_len)
    counts = out.groupby("track_id").size()
    assert (counts >= min_len).all()


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=30, deadline=None)
def test_filter_is_subset(rows):
    df = _make_df(rows)
    out = filter_tracks(df, min_conf=0.0, min_len=1)
    input_pairs = set(zip(df["track_id"], df["frame_idx"]))
    output_pairs = set(zip(out["track_id"], out["frame_idx"]))
    assert output_pairs.issubset(input_pairs)


@pytest.mark.hypothesis
@given(
    rows=track_frame(),
    lo=st.floats(0.0, 0.5),
    hi=st.floats(0.5, 1.0),
)
@settings(max_examples=30, deadline=None)
def test_filter_monotone_in_min_conf(rows, lo, hi):
    df = _make_df(rows)
    out_lo = filter_tracks(df, min_conf=lo, min_len=1)
    out_hi = filter_tracks(df, min_conf=hi, min_len=1)
    lo_pairs = set(zip(out_lo["track_id"], out_lo["frame_idx"]))
    hi_pairs = set(zip(out_hi["track_id"], out_hi["frame_idx"]))
    assert hi_pairs.issubset(lo_pairs)


# --- add_world_coords -------------------------------------------


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=30, deadline=None)
def test_add_world_preserves_row_count(rows):
    df = _make_df(rows)
    out = add_world_coords(df, H_SCALE, fps=30.0)
    assert len(out) == len(df)


@pytest.mark.hypothesis
@given(rows=track_frame(), fps=st.floats(10.0, 60.0))
@settings(max_examples=30, deadline=None)
def test_add_world_t_sec_derived_from_frame_idx(rows, fps):
    df = _make_df(rows)
    if len(df) == 0:
        pytest.skip("empty frame")
    out = add_world_coords(df, H_SCALE, fps=fps)
    expected = out["frame_idx"] / fps
    np.testing.assert_allclose(out["t_sec"].to_numpy(), expected.to_numpy())


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=30, deadline=None)
def test_add_world_speed_is_norm_of_v(rows):
    df = _make_df(rows)
    if len(df) == 0:
        pytest.skip("empty frame")
    out = add_world_coords(df, H_SCALE, fps=30.0)
    expected = np.hypot(out["vx"].to_numpy(), out["vy"].to_numpy())
    np.testing.assert_allclose(out["speed_mps"].to_numpy(), expected)


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=30, deadline=None)
def test_add_world_first_frame_of_track_has_zero_velocity(rows):
    df = _make_df(rows)
    if len(df) == 0:
        pytest.skip("empty frame")
    out = add_world_coords(df, H_SCALE, fps=30.0)
    for tid, group in out.groupby("track_id"):
        first = group.sort_values("frame_idx").iloc[0]
        assert first["vx"] == 0.0, f"track {tid} first vx={first['vx']}"
        assert first["vy"] == 0.0, f"track {tid} first vy={first['vy']}"


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=30, deadline=None)
def test_add_world_identity_homography(rows):
    df = _make_df(rows)
    if len(df) == 0:
        pytest.skip("empty frame")
    out = add_world_coords(df, np.eye(3), fps=30.0)
    np.testing.assert_allclose(
        out["x_m"].to_numpy(), out["x_center"].to_numpy(), atol=1e-9
    )
    np.testing.assert_allclose(
        out["y_m"].to_numpy(), out["y_bottom"].to_numpy(), atol=1e-9
    )


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=30, deadline=None)
def test_add_world_is_deterministic(rows):
    df = _make_df(rows)
    if len(df) == 0:
        pytest.skip("empty frame")
    a = add_world_coords(df, H_SCALE, fps=30.0)
    b = add_world_coords(df, H_SCALE, fps=30.0)
    pd.testing.assert_frame_equal(a, b)


# --- load_tracks -----------------------------------------------


@pytest.mark.hypothesis
@given(rows=track_frame())
@settings(max_examples=30, deadline=None)
def test_load_tracks_roundtrip(rows):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tracks.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + chr(10))
        loaded = load_tracks(path)
    assert len(loaded) == len(rows)
