import numpy as np
import pytest

from graf.calibration.homography import (
    HomographyResult,
    fit_homography,
    invert_homography,
    project_points,
    world_to_image,
)


def test_project_points_basic():
    H = np.eye(3)
    pts = [(1.0, 2.0), (3.0, 4.0)]
    result = project_points(H, pts)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(result, expected)


def test_project_points_zero_scale():
    H = np.zeros((3, 3))
    with pytest.raises(ZeroDivisionError):
        project_points(H, [(1.0, 1.0)])


def test_fit_homography_identity():
    world = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    image = world.copy()
    res = fit_homography(image, world)
    assert isinstance(res, HomographyResult)
    assert np.allclose(res.H, np.eye(3))
    assert res.rms_error < 1e-6


def test_fit_homography_insufficient_points():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2)])


def test_fit_homography_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 1)])


def test_invert_homography_singular():
    H = np.zeros((3, 3))
    with pytest.raises(RuntimeError):
        invert_homography(H)


def test_world_to_image_roundtrip():
    H = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]], dtype=np.float64)
    world = [(1.0, 1.0), (2.0, 3.0)]
    img = world_to_image(H, world)
    back = project_points(H, [tuple(p) for p in img])
    assert np.allclose(back, world)
