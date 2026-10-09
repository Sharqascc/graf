"""Property-based tests for ``graf.graph.builders``.

Complements ``tests/test_property_based.py``, which already asserts node
count, feature dimensions, edge-index bounds, self-loop policy, directed
symmetry, and reverse-edge feature mirroring. The invariants here are
the ones the existing file does *not* cover:

* build determinism (two calls with the same input agree exactly),
* empty-actor handling,
* monotonicity of the edge set in the interaction radius,
* ``one_hot_actor_class`` contract,
* ``get_pair_radius`` order-independence and default fallback.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

builders = pytest.importorskip("graf.graph.builders")
GraphBuilder = builders.GraphBuilder
one_hot_actor_class = builders.one_hot_actor_class
get_pair_radius = builders.get_pair_radius
ACTOR_CLASSES = list(builders.ACTOR_CLASSES)
DEFAULT_RADII = builders.DEFAULT_INTERACTION_RADII

_KNOWN_CLASSES = frozenset(c for pair in DEFAULT_RADII for c in pair)

_coord = st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False)
_vel = st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False)
_class = st.sampled_from(ACTOR_CLASSES)


@st.composite
def actor(draw):
    return {
        "x_m": draw(_coord),
        "y_m": draw(_coord),
        "vx": draw(_vel),
        "vy": draw(_vel),
        "class_name": draw(_class),
        "track_id": draw(st.integers(0, 1000)),
    }


@st.composite
def actor_lists(draw, n_min=1, n_max=8):
    return draw(st.lists(actor(), min_size=n_min, max_size=n_max))


# ── Determinism ──────────────────────────────────────────────────────
@pytest.mark.hypothesis
@given(actors=actor_lists())
@settings(max_examples=30, deadline=None)
def test_graph_build_is_deterministic(actors):
    gb = GraphBuilder(radius=5.0, directed=True, include_self_loops=False)
    a = gb.build_pyg_data(actors)
    b = gb.build_pyg_data(actors)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.equal(a.edge_attr, b.edge_attr)


# ── Empty input ──────────────────────────────────────────────────────
@pytest.mark.hypothesis
@given(r=st.floats(1.0, 10.0))
@settings(max_examples=10, deadline=None)
def test_graph_build_empty_input_returns_empty_graph(r):
    data = GraphBuilder(radius=r).build_pyg_data([])
    assert data.num_nodes == 0
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape[0] == 0


# ── Edge set is monotone in the interaction radius ───────────────────
@pytest.mark.hypothesis
@given(
    actors=actor_lists(),
    r_small=st.floats(1.0, 5.0),
    r_large=st.floats(15.0, 30.0),
)
@settings(max_examples=30, deadline=None)
def test_edge_set_monotone_in_pair_radius(actors, r_small, r_large):
    """If two actors are linked at radius r_small, they are linked at
    every larger radius. Uses a single class for all actors so the pair
    radius is controlled directly.
    """
    assume(r_small < r_large)
    for a in actors:
        a["class_name"] = "car"

    small = GraphBuilder(
        directed=True,
        pair_radii={("car", "car"): r_small},
        include_self_loops=False,
        use_class_specific_radii=True,
    ).build_pyg_data(actors)
    large = GraphBuilder(
        directed=True,
        pair_radii={("car", "car"): r_large},
        include_self_loops=False,
        use_class_specific_radii=True,
    ).build_pyg_data(actors)

    es = set(map(tuple, small.edge_index.t().tolist()))
    el = set(map(tuple, large.edge_index.t().tolist()))
    assert es.issubset(el), "edge set must be monotone in interaction radius"


# ── one_hot_actor_class ──────────────────────────────────────────────
@pytest.mark.hypothesis
@given(cls=_class)
@settings(max_examples=20, deadline=None)
def test_one_hot_known_class_sums_to_one(cls):
    vec = one_hot_actor_class(cls)
    assert sum(vec) == pytest.approx(1.0)
    assert set(vec).issubset({0.0, 1.0})


@pytest.mark.hypothesis
@given(name=st.text(min_size=1, max_size=10))
@settings(max_examples=20, deadline=None)
def test_one_hot_any_name_sums_to_one(name):
    """Known or unknown, a one-hot vector always sums to 1."""
    vec = one_hot_actor_class(name)
    assert sum(vec) == pytest.approx(1.0)
    assert set(vec).issubset({0.0, 1.0})


# ── get_pair_radius ──────────────────────────────────────────────────
@pytest.mark.hypothesis
@given(a=_class, b=_class)
@settings(max_examples=40, deadline=None)
def test_get_pair_radius_order_independent(a, b):
    r1 = get_pair_radius(a, b, default_radius=999.0)
    r2 = get_pair_radius(b, a, default_radius=999.0)
    assert r1 == r2


@pytest.mark.hypothesis
@given(
    unknown=st.text(min_size=1, max_size=8).filter(
        lambda s: s not in _KNOWN_CLASSES
    ),
    default=st.floats(0.1, 100.0),
)
@settings(max_examples=20, deadline=None)
def test_get_pair_radius_unknown_pair_returns_default(unknown, default):
    assert get_pair_radius(unknown, unknown, default_radius=default) == default
