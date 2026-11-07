"""Metric grid overlay for homography QA.

Draws a grid of known world-space spacing onto a frame. The grid is a
visualization aid: a human looks at the result and can tell at a glance
whether the pixel -> world homography lands on the actual road surface,
or whether the calibration is skewed, mirrored, or off by a metre.

The grid does not feed any model. It exists to make homography errors
visible before any downstream measurement uses the homography.

Homography convention
---------------------
``H`` maps pixel coordinates to world coordinates (metres), matching
``graf.calibration.homography.project_points``. This module inverts
``H`` internally to draw world-grid lines onto pixel space.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Number of samples along each grid line in world space. 64 is
#: enough that a 720p frame draws smooth lines, and low enough that
#: a 20x20 grid of lines is a few thousand points total.
_LINE_SAMPLES = 64


def invert_homography(H: np.ndarray) -> np.ndarray:
    """Return the 3x3 inverse of a homography.

    Raises ``ValueError`` on a non-3x3 matrix. Raises
    ``np.linalg.LinAlgError`` on a singular matrix; the caller is
    expected to have validated that ``H`` is a real projective
    transform before calling.
    """
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        raise ValueError(f"H must be 3x3, got {H.shape}")
    return np.linalg.inv(H)


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to N 2D points.

    ``points`` is (N, 2) or a single (2,) array. Returns (N, 2).
    Points whose homogeneous w-coordinate is zero map to infinity
    and are returned as NaN rows so the caller can drop them.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    homog = np.hstack([pts, ones])
    out = homog @ H.T
    w = out[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = out[:, :2] / w
    xy[w[:, 0] == 0] = np.nan
    return xy


def _any_in_bounds(pix: np.ndarray, w_img: int, h_img: int) -> bool:
    in_x = (pix[:, 0] >= 0.0) & (pix[:, 0] < float(w_img))
    in_y = (pix[:, 1] >= 0.0) & (pix[:, 1] < float(h_img))
    return bool((in_x & in_y).any())


def compute_grid_lines(
    image_shape: Sequence[int],
    H: np.ndarray,
    spacing_m: float = 5.0,
    extent_m: float = 100.0,
    line_samples: int = _LINE_SAMPLES,
) -> list[np.ndarray]:
    """Compute pixel-space polylines for a metric grid.

    Parameters
    ----------
    image_shape
        ``(H, W)`` or ``(H, W, C)``. Only the first two are used.
    H
        Pixel -> world homography (matches
        ``graf.calibration.homography.project_points``).
    spacing_m
        Distance between adjacent grid lines, in metres.
    extent_m
        Half-extent of the world-space sampling region. Grid lines
        span ``[-extent_m, +extent_m]`` on each axis. Only lines that
        land partially inside the image are returned.
    line_samples
        Number of world-space samples per grid line.

    Returns
    -------
    list[np.ndarray]
        One ``(N, 2)`` float array per visible grid line. Empty if
        no line lands in bounds.
    """
    if spacing_m <= 0:
        raise ValueError(f"spacing_m must be > 0, got {spacing_m}")
    if extent_m <= 0:
        raise ValueError(f"extent_m must be > 0, got {extent_m}")

    h_img = int(image_shape[0])
    w_img = int(image_shape[1])

    H_inv = invert_homography(H)

    grid_axis = np.arange(-extent_m, extent_m + spacing_m, spacing_m)
    line_axis = np.linspace(-extent_m, extent_m, line_samples)

    polylines: list[np.ndarray] = []

    for x in grid_axis:
        world = np.column_stack([np.full_like(line_axis, x), line_axis])
        pix = apply_homography(H_inv, world)
        pix = pix[np.isfinite(pix).all(axis=1)]
        if len(pix) >= 2 and _any_in_bounds(pix, w_img, h_img):
            polylines.append(pix)

    for y in grid_axis:
        world = np.column_stack([line_axis, np.full_like(line_axis, y)])
        pix = apply_homography(H_inv, world)
        pix = pix[np.isfinite(pix).all(axis=1)]
        if len(pix) >= 2 and _any_in_bounds(pix, w_img, h_img):
            polylines.append(pix)

    return polylines


def draw_metric_grid(
    image: np.ndarray,
    H: np.ndarray,
    spacing_m: float = 5.0,
    extent_m: float = 100.0,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Return a copy of ``image`` with a metric grid drawn on it.

    ``image`` must be a BGR ``uint8`` array (as returned by
    ``cv2.imread``). ``H`` is the pixel -> world homography.

    Raises ``ValueError`` on non-uint8 or non-BGR input.
    """
    import cv2

    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image shape must be (H, W, 3), got {image.shape}")

    out = image.copy()
    polylines = compute_grid_lines(
        image_shape=image.shape,
        H=H,
        spacing_m=spacing_m,
        extent_m=extent_m,
    )
    for line in polylines:
        pts = line.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            out, [pts], isClosed=False, color=color, thickness=thickness
        )
    return out
