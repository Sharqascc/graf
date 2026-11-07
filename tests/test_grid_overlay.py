"""Tests for graf.calibration.grid_overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from graf.calibration.grid_overlay import (
    apply_homography,
    compute_grid_lines,
    draw_metric_grid,
    invert_homography,
)


# --- invert_homography -------------------------------------------------


def test_invert_homography_identity():
    inv = invert_homography(np.eye(3))
    np.testing.assert_allclose(inv, np.eye(3), atol=1e-12)


def test_invert_homography_rejects_wrong_shape():
    with pytest.raises(ValueError, match="3x3"):
        invert_homography(np.eye(2))


def test_invert_homography_raises_on_singular():
    with pytest.raises(np.linalg.LinAlgError):
        invert_homography(np.zeros((3, 3)))


# --- apply_homography --------------------------------------------------


def test_apply_homography_identity_is_identity():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = apply_homography(np.eye(3), pts)
    np.testing.assert_allclose(out, pts, atol=1e-12)


def test_apply_homography_scale():
    H = np.diag([2.0, 2.0, 1.0])
    out = apply_homography(H, np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(out, [[2.0, 2.0]], atol=1e-12)


def test_apply_homography_translation():
    H = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]
    )
    out = apply_homography(H, np.array([[0.0, 0.0]]))
    np.testing.assert_allclose(out, [[10.0, 20.0]], atol=1e-12)


def test_apply_homography_single_point_1d_input():
    out = apply_homography(np.eye(3), np.array([5.0, 6.0]))
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[5.0, 6.0]], atol=1e-12)


# --- compute_grid_lines ------------------------------------------------


def test_compute_grid_lines_identity_has_vertical_and_horizontal():
    lines = compute_grid_lines(
        image_shape=(100, 100, 3),
        H=np.eye(3),
        spacing_m=5.0,
        extent_m=100.0,
    )
    assert len(lines) > 0
    vertical = [ln for ln in lines if np.allclose(ln[:, 0], ln[0, 0])]
    horizontal = [ln for ln in lines if np.allclose(ln[:, 1], ln[0, 1])]
    assert len(vertical) >= 10
    assert len(horizontal) >= 10


def test_compute_grid_lines_out_of_bounds_returns_empty():
    H = np.array(
        [[1.0, 0.0, 10000.0], [0.0, 1.0, 10000.0], [0.0, 0.0, 1.0]]
    )
    lines = compute_grid_lines(
        image_shape=(100, 100, 3), H=H, spacing_m=5.0, extent_m=100.0
    )
    assert lines == []


def test_compute_grid_lines_rejects_bad_spacing():
    with pytest.raises(ValueError, match="spacing_m"):
        compute_grid_lines((100, 100, 3), np.eye(3), spacing_m=0.0)


def test_compute_grid_lines_rejects_bad_extent():
    with pytest.raises(ValueError, match="extent_m"):
        compute_grid_lines((100, 100, 3), np.eye(3), extent_m=-1.0)


def test_compute_grid_lines_finer_spacing_gives_more_lines():
    coarse = compute_grid_lines((200, 200, 3), np.eye(3), spacing_m=20.0)
    fine = compute_grid_lines((200, 200, 3), np.eye(3), spacing_m=5.0)
    assert len(fine) > len(coarse)


def test_compute_grid_lines_monotone_in_extent():
    small = compute_grid_lines((100, 100, 3), np.eye(3), spacing_m=5.0,
                               extent_m=50.0)
    large = compute_grid_lines((100, 100, 3), np.eye(3), spacing_m=5.0,
                               extent_m=100.0)
    assert len(large) >= len(small)


# --- draw_metric_grid --------------------------------------------------


def test_draw_metric_grid_preserves_shape_and_dtype():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_metric_grid(img, np.eye(3), spacing_m=10.0)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_draw_metric_grid_does_not_mutate_input():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_metric_grid(img, np.eye(3), spacing_m=10.0)
    assert img.sum() == 0


def test_draw_metric_grid_draws_something():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_metric_grid(img, np.eye(3), spacing_m=10.0)
    assert out.sum() > 0


def test_draw_metric_grid_rejects_non_uint8():
    img = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        draw_metric_grid(img, np.eye(3), spacing_m=10.0)


def test_draw_metric_grid_rejects_grayscale():
    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        draw_metric_grid(img, np.eye(3), spacing_m=10.0)


def test_draw_metric_grid_deterministic():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    a = draw_metric_grid(img, np.eye(3), spacing_m=10.0)
    b = draw_metric_grid(img, np.eye(3), spacing_m=10.0)
    np.testing.assert_array_equal(a, b)


def test_draw_metric_grid_color_change_changes_pixels():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    green = draw_metric_grid(img, np.eye(3), spacing_m=10.0,
                             color=(0, 255, 0))
    red = draw_metric_grid(img, np.eye(3), spacing_m=10.0, color=(0, 0, 255))
    assert not np.array_equal(green, red)


# --- pipeline integration ---------------------------------------------


def test_inspect_grid_step_points_at_draw_grid_script():
    """The pipeline's inspect-grid step must run draw_grid.py (this PR)."""
    from graf.pipeline import INSPECT_STEPS, Context

    grid_step = next(s for s in INSPECT_STEPS if s.name == "grid")
    ctx = Context(site="sci", video=None, root=Path("/repo"))
    cmds = grid_step.build_commands(ctx)
    assert len(cmds) == 1
    assert cmds[0].script == "draw_grid.py"
