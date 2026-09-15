"""Property-based tests for GRAF core utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_point = st.tuples(
    st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
    st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
)
_positive_fps = st.floats(1.0, 120.0, allow_nan=False, allow_infinity=False)

homography = pytest.importorskip("graf.calibration.homography")


@pytest.mark.hypothesis
@given(points=st.lists(_point, min_size=1, max_size=50))
@settings(max_examples=50, deadline=None)
def test_project_points_preserves_count(points):
    out = homography.project_points(np.eye(3), points)
    assert out.shape == (len(points), 2)


@pytest.mark.hypothesis
@given(points=st.lists(_point, min_size=1, max_size=20))
@settings(max_examples=30, deadline=None)
def test_identity_homography_is_identity_map(points):
    projected = homography.project_points(np.eye(3), points)
    for (x, y), (px, py) in zip(points, projected, strict=True):
        assert math.isclose(x, px, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(y, py, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.hypothesis
@given(
    a=st.floats(0.5, 2.0, allow_nan=False, allow_infinity=False),
    b=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    tx=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    ty=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_invert_is_involution(a, b, tx, ty):
    if abs(a * a - b * b) < 1e-6:
        pytest.skip("singular affine matrix")
    H = np.array([[a, b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_back = homography.invert_homography(homography.invert_homography(H))
    np.testing.assert_allclose(H, H_back, rtol=1e-6, atol=1e-6)


kinematics = pytest.importorskip("graf.trajectories.kinematics")


def _make_tracks(n_frames: int, fps: float) -> pd.DataFrame:
    frames = np.arange(n_frames, dtype=float)
    t = frames / fps
    return pd.DataFrame(
        {
            "track_id": np.ones(n_frames, dtype=int),
            "frame_idx": frames,
            "x_m": t,
            "y_m": 0.5 * t,
        }
    )


@pytest.mark.hypothesis
@given(n_frames=st.integers(2, 200), fps=_positive_fps)
@settings(max_examples=30, deadline=None)
def test_speed_is_non_negative(n_frames, fps):
    out = kinematics.compute_kinematics(_make_tracks(n_frames, fps), fps=fps)
    speeds = out["speed_mps"].dropna()
    assert (speeds >= 0).all()


@pytest.mark.hypothesis
@given(n_frames=st.integers(2, 200), fps=_positive_fps)
@settings(max_examples=30, deadline=None)
def test_no_infinities_in_kinematics(n_frames, fps):
    out = kinematics.compute_kinematics(_make_tracks(n_frames, fps), fps=fps)
    for col in ("speed_mps", "vx_mps", "vy_mps", "accel_mps2"):
        assert not np.isinf(out[col].to_numpy(dtype=float)).any(), col


@pytest.mark.hypothesis
@given(n_frames=st.integers(2, 200), fps=_positive_fps)
@settings(max_examples=30, deadline=None)
def test_time_seconds_matches_frame_over_fps(n_frames, fps):
    out = kinematics.compute_kinematics(_make_tracks(n_frames, fps), fps=fps)
    expected = out["frame_idx"].astype(float) / fps
    np.testing.assert_allclose(out["t_sec"].to_numpy(), expected.to_numpy())


# ------------------------------------------------------------------
# TTC — invariants of compute_ttc_constant_velocity
# ------------------------------------------------------------------

ttc = pytest.importorskip("graf.ssm.ttc")

_xy = st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False)
_vel = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
_pt2 = st.tuples(_xy, _xy)
_vel2 = st.tuples(_vel, _vel)


def _ttc(p1, v1, p2, v2, min_distance=1.5):
    return ttc.compute_ttc_constant_velocity(
        np.asarray(p1, float),
        np.asarray(v1, float),
        np.asarray(p2, float),
        np.asarray(v2, float),
        min_distance=min_distance,
    )


@pytest.mark.hypothesis
@given(p1=_pt2, v1=_vel2, p2=_pt2, v2=_vel2)
@settings(max_examples=50, deadline=None)
def test_ttc_non_negative_or_infinite(p1, v1, p2, v2):
    """ttc_seconds is positive-finite or +inf — never negative, never NaN."""
    r = _ttc(p1, v1, p2, v2)
    assert not math.isnan(r.ttc_seconds)
    assert r.ttc_seconds > 0 or math.isinf(r.ttc_seconds)


@pytest.mark.hypothesis
@given(p1=_pt2, v1=_vel2, p2=_pt2, v2=_vel2)
@settings(max_examples=50, deadline=None)
def test_ttc_symmetric_in_actors(p1, v1, p2, v2):
    """Swapping the two actors leaves ttc_seconds unchanged."""
    a = _ttc(p1, v1, p2, v2)
    b = _ttc(p2, v2, p1, v1)
    if math.isinf(a.ttc_seconds) or math.isinf(b.ttc_seconds):
        assert math.isinf(a.ttc_seconds) and math.isinf(b.ttc_seconds)
    else:
        assert math.isclose(a.ttc_seconds, b.ttc_seconds, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.hypothesis
@given(
    p1=_pt2,
    v1=_vel2,
    p2=_pt2,
    v2=_vel2,
    k=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_ttc_scales_with_spatial_unit(p1, v1, p2, v2, k):
    """Scaling positions and min_distance by k scales ttc_seconds by k."""
    base = _ttc(p1, v1, p2, v2, min_distance=1.5)
    scaled = _ttc(
        (p1[0] * k, p1[1] * k), v1, (p2[0] * k, p2[1] * k), v2, min_distance=1.5 * k
    )
    if math.isinf(base.ttc_seconds):
        assert math.isinf(scaled.ttc_seconds)
    else:
        assert math.isclose(scaled.ttc_seconds, base.ttc_seconds * k, rel_tol=1e-6)


@pytest.mark.hypothesis
@given(
    p1=_pt2,
    v1=_vel2,
    p2=_pt2,
    v2=_vel2,
    k=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_ttc_scales_inverse_with_velocity_unit(p1, v1, p2, v2, k):
    """Scaling velocities by k scales ttc_seconds by 1/k."""
    rv = np.asarray(v2, float) - np.asarray(v1, float)
    if float(rv @ rv) < 1e-10:
        return
    base = _ttc(p1, v1, p2, v2)
    scaled = _ttc(p1, (v1[0] * k, v1[1] * k), p2, (v2[0] * k, v2[1] * k))
    if math.isinf(base.ttc_seconds):
        assert math.isinf(scaled.ttc_seconds)
    else:
        assert math.isclose(scaled.ttc_seconds, base.ttc_seconds / k, rel_tol=1e-6)


@pytest.mark.hypothesis
@given(
    x_gap=st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False),
    v_away=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_ttc_diverging_gives_infinite(x_gap, v_away):
    """Actor 2 sits at +x and moves further +x — no collision course."""
    r = _ttc((0.0, 0.0), (0.0, 0.0), (x_gap, 0.0), (v_away, 0.0))
    assert math.isinf(r.ttc_seconds)
    assert r.status == "diverging_or_parallel"
    assert r.is_approaching is False


@pytest.mark.hypothesis
@given(p1=_pt2, v1=_vel2, p2=_pt2, v2=_vel2)
@settings(max_examples=50, deadline=None)
def test_ttc_severity_bounded(p1, v1, p2, v2):
    """severity is always in [0, 1]."""
    r = _ttc(p1, v1, p2, v2)
    assert 0.0 <= r.severity <= 1.0


# ------------------------------------------------------------------
# Graph builder — structural invariants
# ------------------------------------------------------------------

builders = pytest.importorskip("graf.graph.builders")
edges_mod = pytest.importorskip("graf.graph.edges")

_actor_spec = st.tuples(
    st.integers(0, 10),
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(["car", "pedestrian", "two_wheeler"]),
)


def _actors(specs):
    return [
        {"track_id": tid, "x_m": x, "y_m": y, "vx": vx, "vy": vy, "actor_class": cls}
        for (tid, x, y, vx, vy, cls) in specs
    ]


@pytest.mark.hypothesis
@given(actors=st.lists(_actor_spec, min_size=1, max_size=8))
@settings(max_examples=30, deadline=None)
def test_graph_node_count_matches_actors(actors):
    specs = list(actors)
    data = builders.build_pyg_graph_for_frame(_actors(specs))
    assert data.num_nodes == len(specs)
    assert data.x.shape[0] == len(specs)
    assert data.pos.shape[0] == len(specs)


@pytest.mark.hypothesis
@given(actors=st.lists(_actor_spec, min_size=1, max_size=8))
@settings(max_examples=30, deadline=None)
def test_graph_feature_dims(actors):
    data = builders.build_pyg_graph_for_frame(_actors(list(actors)))
    assert data.x.shape[1] == 11 + len(edges_mod.ACTOR_CLASSES)
    assert data.edge_attr.shape[1] == edges_mod.edge_feature_dim()
    assert edges_mod.edge_feature_dim() == 15


@pytest.mark.hypothesis
@given(actors=st.lists(_actor_spec, min_size=2, max_size=8))
@settings(max_examples=30, deadline=None)
def test_graph_edges_have_valid_indices(actors):
    data = builders.build_pyg_graph_for_frame(_actors(list(actors)))
    if data.edge_index.numel() == 0:
        return
    assert int(data.edge_index.min()) >= 0
    assert int(data.edge_index.max()) < data.num_nodes


@pytest.mark.hypothesis
@given(actors=st.lists(_actor_spec, min_size=2, max_size=8))
@settings(max_examples=30, deadline=None)
def test_graph_no_self_loops_by_default(actors):
    data = builders.build_pyg_graph_for_frame(_actors(list(actors)))
    if data.edge_index.numel() == 0:
        return
    assert not bool((data.edge_index[0] == data.edge_index[1]).any())


@pytest.mark.hypothesis
@given(actors=st.lists(_actor_spec, min_size=2, max_size=6))
@settings(max_examples=25, deadline=None)
def test_graph_directed_edges_are_symmetric(actors):
    """directed=True means every (i,j) has a mirrored (j,i)."""
    data = builders.build_pyg_graph_for_frame(_actors(list(actors)))
    if data.edge_index.numel() == 0:
        return
    pairs = {(int(a), int(b)) for a, b in data.edge_index.t().tolist()}
    for i, j in list(pairs):
        assert (j, i) in pairs


@pytest.mark.hypothesis
@given(actors=st.lists(_actor_spec, min_size=2, max_size=4))
@settings(max_examples=20, deadline=None)
def test_graph_reverse_edges_have_mirrored_features(actors):
    """Reversed edge features follow the documented sign rules."""
    data = builders.build_pyg_graph_for_frame(_actors(list(actors)))
    if data.edge_index.numel() == 0:
        return

    by_pair = {
        (int(data.edge_index[0, k]), int(data.edge_index[1, k])): data.edge_attr[
            k
        ].tolist()
        for k in range(data.edge_index.shape[1])
    }
    idx = {n: i for i, n in enumerate(edges_mod.EDGE_FEATURE_ORDER)}

    preserved = (
        "distance",
        "relative_speed",
        "closing_speed",
        "ttc",
        "same_class",
        "rel_heading_cos",
    )
    flipped = (
        "dx",
        "dy",
        "dvx",
        "dvy",
        "bearing_sin",
        "bearing_cos",
        "rel_heading_sin",
    )

    for (i, j), f_ij in list(by_pair.items()):
        if i == j or (j, i) not in by_pair:
            continue
        f_ji = by_pair[(j, i)]
        for name in preserved:
            assert math.isclose(
                f_ij[idx[name]], f_ji[idx[name]], rel_tol=1e-4, abs_tol=1e-6
            ), name
        for name in flipped:
            assert math.isclose(
                f_ij[idx[name]], -f_ji[idx[name]], rel_tol=1e-4, abs_tol=1e-6
            ), name
        assert math.isclose(f_ij[idx["size_src"]], f_ji[idx["size_dst"]], rel_tol=1e-6)
        assert math.isclose(f_ij[idx["size_dst"]], f_ji[idx["size_src"]], rel_tol=1e-6)
