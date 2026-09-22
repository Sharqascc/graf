"""Property-based tests for GRAF core utilities."""

from __future__ import annotations

import itertools
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
    # 0.0 is valid: "already in collision" at t=0.
    assert r.ttc_seconds >= 0 or math.isinf(r.ttc_seconds)


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
    x_gap=st.floats(2.0, 100.0, allow_nan=False, allow_infinity=False),
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


# ------------------------------------------------------------------
# PET — invariants of compute_pet_from_conflict_zone
# ------------------------------------------------------------------

pet = pytest.importorskip("graf.ssm.pet")

_coord = st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False)
_time_step = st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False)
_radius = st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)


def _single_traj(center, n, dt, offset_x=0.0, offset_y=0.0):
    """Straight-line trajectory passing through center ± offsets."""
    ts = np.arange(n, dtype=float) * dt
    xs = center[0] + offset_x + np.linspace(-1.0, 1.0, n)
    ys = center[1] + offset_y + np.zeros(n)
    return np.stack([xs, ys], axis=1), ts


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 30),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    radius=_radius,
)
@settings(max_examples=40, deadline=None)
def test_pet_status_is_from_known_enum(n, dt, cx, cy, radius):
    """Status is always one of the documented strings."""
    known = {
        "uncomputed",
        "empty_trajectory",
        "one_or_both_never_enter_zone",
        "agent1_then_agent2",
        "agent2_then_agent1",
        "zone_overlap",
    }
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt, offset_x=-0.5)
    t2, ts2 = _single_traj(center, n, dt, offset_x=+0.5)
    r = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    assert r.status in known, r.status


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 30),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    radius=_radius,
)
@settings(max_examples=40, deadline=None)
def test_pet_seconds_non_negative_when_finite(n, dt, cx, cy, radius):
    """pet_seconds is either +inf or a finite value >= 0."""
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt, offset_x=-0.5)
    t2, ts2 = _single_traj(center, n, dt, offset_x=+0.5)
    r = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    assert not math.isnan(r.pet_seconds)
    if math.isfinite(r.pet_seconds):
        assert r.pet_seconds >= 0.0


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 20),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    radius=_radius,
)
@settings(max_examples=30, deadline=None)
def test_pet_empty_trajectory_status(n, dt, cx, cy, radius):
    """Empty input gives +inf and status empty_trajectory."""
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt)
    empty = np.empty((0, 2), dtype=float)
    r = pet.compute_pet_from_conflict_zone(
        empty, t1, np.empty(0), ts1, center, zone_radius=radius
    )
    assert r.status == "empty_trajectory"
    assert math.isinf(r.pet_seconds)


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 20),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    gap=st.floats(100.0, 1000.0, allow_nan=False, allow_infinity=False),
    radius=_radius,
)
@settings(max_examples=30, deadline=None)
def test_pet_never_enter_zone_gives_infinite(n, dt, cx, cy, gap, radius):
    """Trajectories far from the zone give +inf and never-enter status."""
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt, offset_x=gap)
    t2, ts2 = _single_traj(center, n, dt, offset_x=gap + 1.0)
    r = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    assert r.status == "one_or_both_never_enter_zone"
    assert math.isinf(r.pet_seconds)


@pytest.mark.hypothesis
@given(
    n=st.integers(3, 20),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    radius=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_pet_symmetric_in_actors(n, dt, cx, cy, radius):
    """Swapping the two actors swaps enters_first but not pet_seconds."""
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt, offset_x=-2.0)
    t2, ts2 = _single_traj(center, n, dt, offset_x=+2.0)
    a = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    b = pet.compute_pet_from_conflict_zone(t2, t1, ts2, ts1, center, zone_radius=radius)
    if math.isfinite(a.pet_seconds) and math.isfinite(b.pet_seconds):
        assert math.isclose(a.pet_seconds, b.pet_seconds, rel_tol=1e-9, abs_tol=1e-9)
    # enters_first is either None (overlap) on both sides, or mirrored
    if a.enters_first == "agent1":
        assert b.enters_first == "agent2"
    elif a.enters_first == "agent2":
        assert b.enters_first == "agent1"
    else:
        assert b.enters_first is None


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 20),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    radius=_radius,
)
@settings(max_examples=30, deadline=None)
def test_pet_deterministic(n, dt, cx, cy, radius):
    """Same inputs -> same result."""
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt, offset_x=-0.5)
    t2, ts2 = _single_traj(center, n, dt, offset_x=+0.5)
    a = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    b = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    assert a.status == b.status
    assert a.enters_first == b.enters_first
    if math.isfinite(a.pet_seconds):
        assert a.pet_seconds == b.pet_seconds


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 20),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    radius=_radius,
)
@settings(max_examples=30, deadline=None)
def test_pet_severity_bounded(n, dt, cx, cy, radius):
    """severity is always in [0, 1]."""
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt, offset_x=-0.5)
    t2, ts2 = _single_traj(center, n, dt, offset_x=+0.5)
    r = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    assert 0.0 <= r.severity <= 1.0


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 20),
    dt=_time_step,
    cx=_coord,
    cy=_coord,
    radius=_radius,
)
@settings(max_examples=30, deadline=None)
def test_pet_is_critical_iff_short_and_finite(n, dt, cx, cy, radius):
    """is_critical <=> pet in [0, 3] and finite."""
    center = np.array([cx, cy])
    t1, ts1 = _single_traj(center, n, dt, offset_x=-0.5)
    t2, ts2 = _single_traj(center, n, dt, offset_x=+0.5)
    r = pet.compute_pet_from_conflict_zone(t1, t2, ts1, ts2, center, zone_radius=radius)
    expected = math.isfinite(r.pet_seconds) and 0.0 <= r.pet_seconds <= 3.0
    assert r.is_critical == expected


# ------------------------------------------------------------------
# PETCalculator — invariants of the DataFrame-level API
# ------------------------------------------------------------------

_required_cols = ["track_id", "frame_idx", "t_sec", "x_m", "y_m"]


def _pet_df(track_id, n, dt, x0):
    return pd.DataFrame(
        {
            "track_id": [track_id] * n,
            "frame_idx": list(range(n)),
            "t_sec": [i * dt for i in range(n)],
            "x_m": [x0 + i * 0.1 for i in range(n)],
            "y_m": [0.0] * n,
        }
    )


@pytest.mark.hypothesis
@given(
    n=st.integers(2, 20),
    dt=_time_step,
)
@settings(max_examples=20, deadline=None)
def test_pet_calculator_short_trajectory_returns_none(n, dt):
    """Tracks with <2 rows give None (no PET event)."""
    calc = pet.PETCalculator(proximity_threshold_m=2.0)
    a = pd.DataFrame(
        {"track_id": [0], "frame_idx": [0], "t_sec": [0.0], "x_m": [0.0], "y_m": [0.0]}
    )
    b = _pet_df(1, n, dt, x0=0.0)
    assert calc.compute_pair_pet(a, b, video_id="v") is None
    assert calc.compute_pair_pet(b, a, video_id="v") is None


@pytest.mark.hypothesis
@given(
    n=st.integers(3, 20),
    dt=_time_step,
    gap=st.floats(100.0, 1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_pet_calculator_far_apart_returns_none(n, dt, gap):
    """Pairs whose min distance exceeds the proximity threshold give None."""
    calc = pet.PETCalculator(proximity_threshold_m=2.0)
    a = _pet_df(0, n, dt, x0=0.0)
    b = _pet_df(1, n, dt, x0=gap)
    assert calc.compute_pair_pet(a, b, video_id="v") is None


@pytest.mark.hypothesis
@given(n=st.integers(2, 10), dt=_time_step)
@settings(max_examples=15, deadline=None)
def test_pet_calculator_missing_columns_raises(n, dt):
    """Missing required columns raise ValueError."""
    calc = pet.PETCalculator()
    a = _pet_df(0, n, dt, x0=0.0).drop(columns=["x_m"])
    b = _pet_df(1, n, dt, x0=0.0)
    with pytest.raises(ValueError):
        calc.compute_pair_pet(a, b, video_id="v")


# ------------------------------------------------------------------
# DRAC — invariants of compute_drac_constant_velocity
# ------------------------------------------------------------------

drac_mod = pytest.importorskip("graf.ssm.drac")

_pos2d = st.tuples(
    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
)
_vel2d = st.tuples(
    st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False),
    st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False),
)


def _drac(p1, v1, p2, v2, collision_radius=1.5):
    return drac_mod.compute_drac_constant_velocity(
        np.asarray(p1, float),
        np.asarray(v1, float),
        np.asarray(p2, float),
        np.asarray(v2, float),
        collision_radius=collision_radius,
    )


@pytest.mark.hypothesis
@given(p1=_pos2d, v1=_vel2d, p2=_pos2d, v2=_vel2d)
@settings(max_examples=50, deadline=None)
def test_drac_non_negative_or_infinite(p1, v1, p2, v2):
    """drac_mps2 is always >= 0 (finite) or +inf; never NaN, never negative."""
    r = _drac(p1, v1, p2, v2)
    assert not math.isnan(r.drac_mps2)
    assert r.drac_mps2 >= 0.0 or math.isinf(r.drac_mps2)


@pytest.mark.hypothesis
@given(p1=_pos2d, v1=_vel2d, p2=_pos2d, v2=_vel2d)
@settings(max_examples=50, deadline=None)
def test_drac_symmetric_in_actors(p1, v1, p2, v2):
    """Swapping the two actors leaves DRAC and status unchanged."""
    a = _drac(p1, v1, p2, v2)
    b = _drac(p2, v2, p1, v1)
    assert a.status == b.status
    if math.isinf(a.drac_mps2) or math.isinf(b.drac_mps2):
        assert math.isinf(a.drac_mps2) and math.isinf(b.drac_mps2)
    else:
        assert math.isclose(a.drac_mps2, b.drac_mps2, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(
            a.closing_speed_mps, b.closing_speed_mps, rel_tol=1e-9, abs_tol=1e-9
        )


@pytest.mark.hypothesis
@given(
    p1=_pos2d,
    p2=st.tuples(
        st.floats(2.0, 100.0, allow_nan=False, allow_infinity=False),
        st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
    ),
    base_speed=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
    k=st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_drac_quadratic_in_closing_speed(p1, p2, base_speed, k):
    """Scaling closing speed by k scales DRAC by k^2 (fixed gap)."""
    # Head-on approach along the x-axis: v1 = +base on x, v2 = -base on x
    v1a = (base_speed, 0.0)
    v2a = (-base_speed, 0.0)
    v1b = (base_speed * k, 0.0)
    v2b = (-base_speed * k, 0.0)
    a = _drac(p1, v1a, p2, v2a)
    b = _drac(p1, v1b, p2, v2b)
    if a.status != "computed" or b.status != "computed":
        return
    assert math.isclose(b.drac_mps2, a.drac_mps2 * k * k, rel_tol=1e-6)


@pytest.mark.hypothesis
@given(
    gap1=st.floats(3.0, 100.0, allow_nan=False, allow_infinity=False),
    gap2=st.floats(3.0, 100.0, allow_nan=False, allow_infinity=False),
    speed=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_drac_decreases_with_gap(gap1, gap2, speed):
    """At fixed closing speed, larger gap -> smaller or equal DRAC."""
    # Centres at x = gap + collision_radius so effective gap == gap
    cr = 1.5
    r1 = _drac(
        (0.0, 0.0), (speed, 0.0), (gap1 + cr, 0.0), (-speed, 0.0), collision_radius=cr
    )
    r2 = _drac(
        (0.0, 0.0), (speed, 0.0), (gap2 + cr, 0.0), (-speed, 0.0), collision_radius=cr
    )
    if gap1 < gap2:
        assert r1.drac_mps2 >= r2.drac_mps2 - 1e-9
    elif gap2 < gap1:
        assert r2.drac_mps2 >= r1.drac_mps2 - 1e-9
    else:
        assert math.isclose(r1.drac_mps2, r2.drac_mps2, rel_tol=1e-9)


@pytest.mark.hypothesis
@given(p1=_pos2d, v1=_vel2d, p2=_pos2d, v2=_vel2d)
@settings(max_examples=50, deadline=None)
def test_drac_deterministic(p1, v1, p2, v2):
    """Same inputs -> same result."""
    a = _drac(p1, v1, p2, v2)
    b = _drac(p1, v1, p2, v2)
    assert a.status == b.status
    if math.isfinite(a.drac_mps2) and math.isfinite(b.drac_mps2):
        assert a.drac_mps2 == b.drac_mps2


@pytest.mark.hypothesis
@given(
    gap_override=st.floats(3.0, 200.0, allow_nan=False, allow_infinity=False),
    speed=st.floats(0.5, 10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_drac_gap_override_ignores_positions(gap_override, speed):
    """With gap_override set, positions do not affect the result."""
    # Two wildly different position pairs, same gap_override
    r1 = drac_mod.compute_drac_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([speed, 0.0]),
        np.array([1.0, 0.0]),
        np.array([-speed, 0.0]),
        gap_override=gap_override,
    )
    r2 = drac_mod.compute_drac_constant_velocity(
        np.array([100.0, 100.0]),
        np.array([speed, 0.0]),
        np.array([1.0, 1.0]),
        np.array([-speed, 0.0]),
        gap_override=gap_override,
    )
    if r1.status == "computed" and r2.status == "computed":
        assert r1.gap_m == r2.gap_m
        # Closing speed depends on geometry (unit vector), which changes;
        # but the raw gap must be identical.
        assert r1.gap_m == pytest.approx(gap_override - 1.5)


@pytest.mark.hypothesis
@given(p1=_pos2d, v1=_vel2d, p2=_pos2d, v2=_vel2d)
@settings(max_examples=50, deadline=None)
def test_drac_severity_bounded(p1, v1, p2, v2):
    """severity is always in [0, 1]."""
    r = _drac(p1, v1, p2, v2)
    assert 0.0 <= r.severity <= 1.0


@pytest.mark.hypothesis
@given(p1=_pos2d, v1=_vel2d, p2=_pos2d, v2=_vel2d)
@settings(max_examples=50, deadline=None)
def test_drac_is_critical_iff_finite_and_above_threshold(p1, v1, p2, v2):
    """is_critical <=> drac_mps2 is finite and >= CRITICAL_DRAC_MPS2."""
    r = _drac(p1, v1, p2, v2)
    expected = math.isfinite(r.drac_mps2) and r.drac_mps2 >= drac_mod.CRITICAL_DRAC_MPS2
    assert r.is_critical == expected


# ------------------------------------------------------------------
# Event mining — invariants of mine_events
# ------------------------------------------------------------------

mining_mod = pytest.importorskip("graf.ssm.event_mining")

_thresholds_map = {"TTC": 1.5, "PET": 2.0, "DRAC": 3.35}


@st.composite
def _ssm_dataframe(draw, max_frames=15, max_pairs=2):
    """Build a plausible ssm_frame_values DataFrame."""
    n_pairs = draw(st.integers(1, max_pairs))
    pairs = [(f"a{i}", f"b{i}") for i in range(n_pairs)]
    n_frames = draw(st.integers(1, max_frames))
    metric = draw(st.sampled_from(list(_thresholds_map.keys())))
    rows = []
    for ta, tb in pairs:
        for f in range(n_frames):
            v = draw(st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False))
            rows.append(("v", f, ta, tb, metric, v))
    return pd.DataFrame(
        rows,
        columns=[
            "video_id",
            "frame_idx",
            "track_id_a",
            "track_id_b",
            "metric_name",
            "value",
        ],
    )


@pytest.mark.hypothesis
@given(df=_ssm_dataframe())
@settings(max_examples=25, deadline=None)
def test_event_mining_records_validate(df):
    """Every returned record validates and has the required fields."""
    events = mining_mod.mine_events(df, thresholds=_thresholds_map)
    for e in events:
        e.validate()
        assert e.metric_name in _thresholds_map
        assert e.start_frame <= e.end_frame
        assert e.metadata["num_frames"] >= 1
        assert e.metadata["metric_direction"] in {"below", "above"}


@pytest.mark.hypothesis
@given(df=_ssm_dataframe(), mdf=st.integers(1, 5))
@settings(max_examples=25, deadline=None)
def test_event_mining_respects_min_duration(df, mdf):
    """Every event has at least min_duration_frames rows."""
    events = mining_mod.mine_events(
        df, thresholds=_thresholds_map, min_duration_frames=mdf
    )
    for e in events:
        assert e.metadata["num_frames"] >= mdf


@pytest.mark.hypothesis
@given(df=_ssm_dataframe(), mdf=st.integers(1, 4))
@settings(max_examples=20, deadline=None)
def test_event_mining_events_within_group_disjoint(df, mdf):
    """For a fixed group, event intervals do not overlap."""
    events = mining_mod.mine_events(
        df, thresholds=_thresholds_map, min_duration_frames=mdf
    )
    groups = {}
    for e in events:
        key = (e.video_id, e.track_id_a, e.track_id_b, e.metric_name)
        groups.setdefault(key, []).append(e)
    for group_events in groups.values():
        group_events.sort(key=lambda e: e.start_frame)
        for a, b in itertools.pairwise(group_events):
            assert b.start_frame > a.end_frame


@pytest.mark.hypothesis
@given(df=_ssm_dataframe())
@settings(max_examples=25, deadline=None)
def test_event_mining_extreme_crosses_threshold(df):
    """min_value crosses the threshold in the metric direction."""
    events = mining_mod.mine_events(df, thresholds=_thresholds_map)
    for e in events:
        threshold = _thresholds_map[e.metric_name]
        direction = e.metadata["metric_direction"]
        if direction == "below":
            assert e.min_value <= threshold
        else:
            assert e.min_value >= threshold


@pytest.mark.hypothesis
@given(df=_ssm_dataframe())
@settings(max_examples=20, deadline=None)
def test_event_mining_deterministic(df):
    """Same input -> same output."""
    a = mining_mod.mine_events(df, thresholds=_thresholds_map)
    b = mining_mod.mine_events(df, thresholds=_thresholds_map)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


@pytest.mark.hypothesis
@given(df=_ssm_dataframe(), gap=st.integers(1, 3))
@settings(max_examples=20, deadline=None)
def test_event_mining_frame_gaps_respected(df, gap):
    """No internal frame_idx gap within an event exceeds max_frame_gap."""
    events = mining_mod.mine_events(df, thresholds=_thresholds_map, max_frame_gap=gap)
    for e in events:
        mask = (
            (df["video_id"] == e.video_id)
            & (df["track_id_a"] == e.track_id_a)
            & (df["track_id_b"] == e.track_id_b)
            & (df["metric_name"] == e.metric_name)
            & (df["frame_idx"] >= e.start_frame)
            & (df["frame_idx"] <= e.end_frame)
        )
        frames = sorted(df.loc[mask, "frame_idx"].tolist())
        for a, b in itertools.pairwise(frames):
            assert b - a <= gap


@pytest.mark.hypothesis
@given(df=_ssm_dataframe())
@settings(max_examples=15, deadline=None)
def test_event_mining_empty_thresholds_returns_empty(df):
    assert mining_mod.mine_events(df, thresholds={}) == []


# ------------------------------------------------------------------
# SSM invariants added after the mutation testing diagnostic
# ------------------------------------------------------------------


@given(
    sep=st.floats(min_value=5.0, max_value=100.0, allow_nan=False),
    closing=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
)
def test_ttc_monotone_in_initial_separation(sep, closing) -> None:
    """All else equal, further initial separation -> TTC is weakly larger."""
    from graf.ssm.ttc import compute_ttc_constant_velocity

    pos1 = np.zeros(2)
    vel1 = np.zeros(2)
    # Actor 2 at +sep, closing along -x at `closing` m/s.
    r_near = compute_ttc_constant_velocity(
        pos1,
        vel1,
        np.array([sep, 0.0]),
        np.array([-closing, 0.0]),
        min_distance=1.5,
    )
    r_far = compute_ttc_constant_velocity(
        pos1,
        vel1,
        np.array([sep + 5.0, 0.0]),
        np.array([-closing, 0.0]),
        min_distance=1.5,
    )
    assert r_far.ttc_seconds >= r_near.ttc_seconds, (
        f"further separation gave smaller TTC: far={r_far.ttc_seconds}, "
        f"near={r_near.ttc_seconds}"
    )


@given(
    sep=st.floats(min_value=5.0, max_value=100.0, allow_nan=False),
    v_slow=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
    v_fast=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
)
def test_ttc_monotone_in_closing_speed(sep, v_slow, v_fast) -> None:
    """All else equal, higher closing speed -> TTC is weakly smaller."""
    from graf.ssm.ttc import compute_ttc_constant_velocity

    pos1 = np.zeros(2)
    vel1 = np.zeros(2)
    r_slow = compute_ttc_constant_velocity(
        pos1,
        vel1,
        np.array([sep, 0.0]),
        np.array([-v_slow, 0.0]),
        min_distance=1.5,
    )
    r_fast = compute_ttc_constant_velocity(
        pos1,
        vel1,
        np.array([sep, 0.0]),
        np.array([-v_fast, 0.0]),
        min_distance=1.5,
    )
    assert r_fast.ttc_seconds <= r_slow.ttc_seconds, (
        f"faster closing gave larger TTC: fast={r_fast.ttc_seconds}, "
        f"slow={r_slow.ttc_seconds}"
    )


@given(drac=st.floats(min_value=0.0, max_value=20.0, allow_nan=False))
def test_drac_critical_matches_threshold_constant(drac) -> None:
    """DRACResult.is_critical is a property (no args) keyed to the module
    constant CRITICAL_DRAC_MPS2."""
    from graf.ssm.drac import CRITICAL_DRAC_MPS2, DRACResult

    r = DRACResult(drac_mps2=float(drac))
    if drac >= CRITICAL_DRAC_MPS2:
        assert r.is_critical, f"drac={drac} >= threshold but not critical"
    else:
        assert not r.is_critical, f"drac={drac} < threshold but critical"


@given(
    v_max=st.floats(min_value=1.0, max_value=30.0, allow_nan=False),
    dt=st.floats(min_value=1 / 60.0, max_value=1.0, allow_nan=False),
    n_steps=st.integers(min_value=2, max_value=20),
)
def test_trajectory_physical_bounds(v_max, dt, n_steps) -> None:
    """Constant-velocity trajectory with speed <= v_max satisfies
    |pos[i+1] - pos[i]| <= v_max * dt. A tracker or filter that
    violates this produces edges no real actor could traverse.
    """
    import numpy as np

    rng = np.random.default_rng(abs(hash((v_max, dt, n_steps))) % (2**32))
    speed = rng.uniform(0, v_max)
    heading = rng.uniform(0, 2 * np.pi)
    vel = speed * np.array([np.cos(heading), np.sin(heading)])
    pos = np.zeros(2)
    prev = pos.copy()
    for _ in range(n_steps):
        pos = pos + vel * dt
        step = float(np.linalg.norm(pos - prev))
        assert step <= v_max * dt + 1e-9
        prev = pos.copy()


@given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    y0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    dx=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
    n=st.integers(min_value=3, max_value=10),
)
def test_no_teleporting_within_a_window(x0, y0, dx, n) -> None:
    """No consecutive frame in a window may show an actor jumping by
    more than 10x the median step. Catches reorder / duplicate-frame
    bugs in window construction.
    """
    import numpy as np

    positions = np.array([[x0 + i * dx, y0] for i in range(n)])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    median_step = float(np.median(steps))
    if median_step > 0:
        assert steps.max() <= 10.0 * median_step
