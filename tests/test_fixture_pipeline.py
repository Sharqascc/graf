"""End-to-end pipeline test on the committed synthetic fixture.

The fixture lives at data/fixtures/demo/. It is small, deterministic,
and committed to the repo, so this test runs in CI without the real
VNTraffic data. It asserts the pipeline reaches its terminal state
and writes a well-formed comparison.json.

Do NOT assert on specific metric values: the fixture is deliberately
signal-free (AUC near 0.5). Its purpose is to pin the code path, not
to reproduce the paper numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data" / "fixtures" / "demo"


@pytest.fixture(scope="module")
def fixture_paths():
    tracks = FIXTURE / "tracks.jsonl"
    graphs = FIXTURE / "graphs"
    homo = FIXTURE / "homography.yaml"
    assert tracks.exists(), f"missing fixture: {tracks}"
    assert homo.exists(), f"missing fixture: {homo}"
    assert graphs.is_dir() and any(graphs.glob("*.pt")), (
        f"missing fixture graphs: {graphs}"
    )
    return tracks, graphs, homo


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory, fixture_paths):
    """Run the full harness once on the fixture; return the output dir."""
    tracks, graphs, homo = fixture_paths
    out_dir = tmp_path_factory.mktemp("fixture_out")
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "compare_baselines.py"),
        "--tracks",
        str(tracks),
        "--graphs_dir",
        str(graphs),
        "--homography_config",
        str(homo),
        "--output_dir",
        str(out_dir),
        "--models",
        "majority,logreg,rf,single_feature",
        "--num_folds",
        "3",
        "--split",
        "blocked",
        "--label-source",
        "ttc",
        "--label-strategy",
        "sustained",
        "--run-length",
        "3",
        "--ttc-threshold-seconds",
        "1.5",
        "--ttc-distance-threshold",
        "3.0",
        "--calibrate-threshold",
        "--calibration-objective",
        "accuracy",
    ]
    r = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, (
        f"pipeline failed rc={r.returncode}\n"
        f"stdout tail:\n{r.stdout[-2000:]}\n"
        f"stderr tail:\n{r.stderr[-2000:]}"
    )
    return out_dir


def test_pipeline_writes_comparison_json(pipeline_output):
    path = pipeline_output / "comparison.json"
    assert path.exists(), f"no comparison.json at {path}"
    payload = json.loads(path.read_text())
    assert set(payload.keys()) >= {"setup", "labels", "results"}


def test_pipeline_finds_the_expected_windows(pipeline_output):
    payload = json.loads((pipeline_output / "comparison.json").read_text())
    lbl = payload["labels"]
    assert lbl["num_windows"] == 28
    assert lbl["num_positive"] > 0
    assert lbl["num_positive"] < lbl["num_windows"]


def test_pipeline_runs_every_requested_model(pipeline_output):
    payload = json.loads((pipeline_output / "comparison.json").read_text())
    models = [r["model"] for r in payload["results"]]
    for expected in ("majority", "logreg", "rf", "single_feature"):
        assert expected in models, f"{expected} missing from results: {models}"


def test_pipeline_result_shape(pipeline_output):
    payload = json.loads((pipeline_output / "comparison.json").read_text())
    for r in payload["results"]:
        assert 0.0 <= r["mean_accuracy"] <= 1.0
        assert len(r["fold_negative_count"]) == 3
        if "leakage_raw" in r:
            assert r["leakage_raw"]["total"] >= 0
        if "leakage_after_purge" in r:
            assert r["leakage_after_purge"]["total"] == 0, (
                "purge should leave zero leaked validation windows"
            )
