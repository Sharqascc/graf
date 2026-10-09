"""Structural tests for the SCI second-site recipe.

The recipe is data-only, so the tests are schema checks: step names
and order, the detector config, declared outputs, and internal path
consistency between consecutive steps. This catches drift between
the recipe and the pipeline scripts before the workflow runs.
"""

from __future__ import annotations

from pathlib import Path

from graf.reproduce import load_recipe

_REPO = Path(__file__).resolve().parents[1]
_RECIPE = _REPO / "configs" / "recipes" / "sci_v1.yaml"

_EXPECTED_STEPS = ["extract-frames", "detect", "track", "quality-summary"]


def test_recipe_file_exists():
    assert _RECIPE.exists(), f"recipe missing: {_RECIPE}"


def test_recipe_loads():
    recipe = load_recipe(_RECIPE)
    assert recipe.name == "sci_v1"
    assert recipe.steps, "recipe has no steps"


def test_step_names_and_order():
    recipe = load_recipe(_RECIPE)
    assert [s.name for s in recipe.steps] == _EXPECTED_STEPS


def test_uses_uvh26_detector_config():
    recipe = load_recipe(_RECIPE)
    detect = next(s for s in recipe.steps if s.name == "detect")
    assert "configs/detection/uvh26.yaml" in detect.cmd


def test_every_step_declares_produces():
    recipe = load_recipe(_RECIPE)
    for s in recipe.steps:
        assert s.produces is not None, f"step {s.name!r} missing 'produces'"


def test_extract_uses_bounded_time_window():
    recipe = load_recipe(_RECIPE)
    extract = next(s for s in recipe.steps if s.name == "extract-frames")
    assert "--start-time" in extract.cmd
    assert "--duration" in extract.cmd
    dur_idx = extract.cmd.index("--duration")
    dur = float(extract.cmd[dur_idx + 1])
    # Sanity: the recipe is not meant to process the full 14-minute video.
    assert 0 < dur <= 300, f"duration {dur}s outside (0, 300] sanity range"


def test_detect_reads_extract_output():
    recipe = load_recipe(_RECIPE)
    extract = next(s for s in recipe.steps if s.name == "extract-frames")
    detect = next(s for s in recipe.steps if s.name == "detect")
    idx = detect.cmd.index("--frames_dir")
    frames_dir = Path(detect.cmd[idx + 1])
    assert frames_dir.is_relative_to(extract.produces), (
        f"detect reads {frames_dir}, not under extract produces {extract.produces}"
    )


def test_track_reads_detect_output():
    recipe = load_recipe(_RECIPE)
    detect = next(s for s in recipe.steps if s.name == "detect")
    track = next(s for s in recipe.steps if s.name == "track")
    idx = track.cmd.index("--detections")
    det_file = Path(track.cmd[idx + 1])
    assert det_file.is_relative_to(detect.produces.parent), (
        f"track reads {det_file}, not under detect output dir {detect.produces.parent}"
    )


def test_summary_reads_track_output():
    recipe = load_recipe(_RECIPE)
    track = next(s for s in recipe.steps if s.name == "track")
    summary = next(s for s in recipe.steps if s.name == "quality-summary")
    idx = summary.cmd.index("--tracks")
    trk_file = Path(summary.cmd[idx + 1])
    assert trk_file == track.produces, (
        f"summary reads {trk_file}, expected {track.produces}"
    )
