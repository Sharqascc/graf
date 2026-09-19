"""Determinism tests: identical inputs must give bitwise-identical outputs.

Catches the class of bugs that hide behind randomness:

  * Unseeded numpy / torch RNG calls
  * Hash-based iteration order dependencies (dict / set ordering)
  * Parallelism nondeterminism (thread scheduling, race conditions)
  * Silent float drift in reductions that reorder operations

If any of these fire, the test fails on the *first* call to the function
with the same inputs, not on the tenth — so it runs fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from graf.graph.builders import build_pyg_graph_for_frame
from graf.ssm.drac import compute_drac_constant_velocity
from graf.ssm.ttc import compute_ttc_constant_velocity
from graf.trajectories.kinematics import compute_kinematics

# ------------------------------------------------------------------
# TTC / DRAC
# ------------------------------------------------------------------


def test_ttc_bitwise_deterministic():
    p1 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    p2 = np.array([10.0, 0.0])
    v2 = np.array([-1.0, 0.0])
    a = compute_ttc_constant_velocity(p1, v1, p2, v2)
    b = compute_ttc_constant_velocity(p1, v1, p2, v2)
    assert a.ttc_seconds == b.ttc_seconds
    assert a.status == b.status
    assert a.is_approaching == b.is_approaching


def test_drac_bitwise_deterministic():
    p1 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    p2 = np.array([3.0, 0.0])
    v2 = np.array([-1.0, 0.0])
    a = compute_drac_constant_velocity(p1, v1, p2, v2)
    b = compute_drac_constant_velocity(p1, v1, p2, v2)
    assert a.drac_mps2 == b.drac_mps2
    assert a.status == b.status


def test_ttc_repeated_calls_bit_identical():
    """Run 20 times; every result must byte-match the first."""
    p1 = np.array([0.5, 1.5])
    v1 = np.array([2.3, -0.7])
    p2 = np.array([10.0, 0.0])
    v2 = np.array([-1.1, 0.4])
    first = compute_ttc_constant_velocity(p1, v1, p2, v2).ttc_seconds
    for _ in range(20):
        assert compute_ttc_constant_velocity(p1, v1, p2, v2).ttc_seconds == first


# ------------------------------------------------------------------
# Kinematics
# ------------------------------------------------------------------


def _tracks(n_frames: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "track_id": [1] * n_frames,
            "frame_idx": list(range(n_frames)),
            "x_m": rng.normal(0, 1, n_frames).cumsum(),
            "y_m": rng.normal(0, 1, n_frames).cumsum(),
        }
    )


def test_kinematics_bitwise_deterministic():
    df = _tracks()
    a = compute_kinematics(df.copy(), fps=30.0)
    b = compute_kinematics(df.copy(), fps=30.0)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_kinematics_repeated_calls_bit_identical():
    df = _tracks(n_frames=15)
    base = compute_kinematics(df.copy(), fps=30.0)
    for _ in range(10):
        result = compute_kinematics(df.copy(), fps=30.0)
        pd.testing.assert_frame_equal(base, result, check_exact=True)


# ------------------------------------------------------------------
# Graph builder (torch tensors)
# ------------------------------------------------------------------


def _sample_actors():
    return [
        {
            "track_id": 1,
            "x_m": 0.0,
            "y_m": 0.0,
            "vx": 1.0,
            "vy": 0.0,
            "actor_class": "car",
        },
        {
            "track_id": 2,
            "x_m": 3.0,
            "y_m": 0.0,
            "vx": -1.0,
            "vy": 0.0,
            "actor_class": "car",
        },
    ]


def test_graph_builder_bitwise_deterministic():
    a = build_pyg_graph_for_frame(_sample_actors())
    b = build_pyg_graph_for_frame(_sample_actors())
    torch.testing.assert_close(a.x, b.x, rtol=0, atol=0)
    torch.testing.assert_close(a.edge_index, b.edge_index, rtol=0, atol=0)
    torch.testing.assert_close(a.edge_attr, b.edge_attr, rtol=0, atol=0)
    torch.testing.assert_close(a.pos, b.pos, rtol=0, atol=0)


# ------------------------------------------------------------------
# RNG seed discipline
# ------------------------------------------------------------------


def test_numpy_seed_reproducible():
    """A canary: if this fails, the environment RNG is nondeterministic."""
    np.random.seed(42)
    a = np.random.rand(5)
    np.random.seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_torch_seed_reproducible():
    torch.manual_seed(42)
    a = torch.randn(5)
    torch.manual_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)
