"""Tests for scripts/density_diagnostic.py.

Two layers:

1. Unit tests on the pure helpers (random_folds, evaluate_bin). No I/O.
2. One integration test that runs the CLI end-to-end on the committed
   demo fixture. Proves the argparse surface is stable and the output
   JSON has the expected schema, without needing real data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.density_diagnostic import (
    evaluate_bin,
    random_folds,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data" / "fixtures" / "demo"


# ---------- random_folds ----------


def test_random_folds_covers_every_index_once():
    n, k = 100, 5
    folds = random_folds(n, k, seed=42)
    assert len(folds) == k
    seen = np.concatenate([val for _, val in folds])
    assert sorted(seen.tolist()) == list(range(n))


def test_random_folds_train_val_disjoint():
    for tr, va in random_folds(60, 4, seed=7):
        assert len(np.intersect1d(tr, va)) == 0


def test_random_folds_deterministic_for_fixed_seed():
    a = random_folds(40, 4, seed=42)
    b = random_folds(40, 4, seed=42)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b, strict=True):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


# ---------- evaluate_bin ----------


def test_evaluate_bin_skipped_when_all_one_class():
    X = np.random.RandomState(0).randn(50, 5)
    y = np.ones(50, dtype=int)
    out = evaluate_bin(X, y, k=5, seed=0)
    assert out["skipped"] is True
    assert out["n"] == 50


def test_evaluate_bin_skipped_when_too_small():
    X = np.random.RandomState(0).randn(4, 3)
    y = np.array([0, 1, 0, 1])
    out = evaluate_bin(X, y, k=5, seed=0)
    assert out["skipped"] is True


def test_evaluate_bin_runs_on_balanced_binary():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 4)
    # signal: y depends on X[:, 0]
    y = (X[:, 0] + 0.3 * rng.randn(n) > 0).astype(int)
    out = evaluate_bin(X, y, k=5, seed=42)
    assert out["skipped"] is False
    assert out["n"] == n
    # both baselines ran
    assert out["rf"]["mean_auc"] is not None
    assert out["single_feature"]["mean_auc"] is not None
    # with a real signal, RF should beat chance
    assert out["rf"]["mean_auc"] > 0.55


def test_evaluate_bin_rf_auc_bounded():
    rng = np.random.RandomState(7)
    n = 100
    X = rng.randn(n, 3)
    y = rng.randint(0, 2, n)
    out = evaluate_bin(X, y, k=5, seed=42)
    if not out["skipped"] and out["rf"]["mean_auc"] is not None:
        assert 0.0 <= out["rf"]["mean_auc"] <= 1.0


# ---------- CLI integration on demo fixture ----------


@pytest.fixture(scope="module")
def fixture_paths():
    tracks = FIXTURE / "tracks.jsonl"
    graphs = FIXTURE / "graphs"
    homo = FIXTURE / "homography.yaml"
    assert tracks.exists(), f"missing fixture: {tracks}"
    assert graphs.is_dir() and any(graphs.glob("*.pt")), (
        f"missing fixture graphs: {graphs}"
    )
    assert homo.exists(), f"missing fixture: {homo}"
    return tracks, graphs, homo


def test_cli_runs_end_to_end_on_fixture(tmp_path, fixture_paths):
    tracks, graphs, homo = fixture_paths
    out_dir = tmp_path / "density_out"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "density_diagnostic.py"),
        "--tracks",
        str(tracks),
        "--graphs_dir",
        str(graphs),
        "--homography_config",
        str(homo),
        "--label_strategy",
        "any",
        "--output_dir",
        str(out_dir),
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (
        f"CLI failed rc={r.returncode}\n"
        f"stdout tail:\n{r.stdout[-1500:]}\n"
        f"stderr tail:\n{r.stderr[-1500:]}"
    )
    json_path = out_dir / "density_results.json"
    md_path = out_dir / "density_results.md"
    assert json_path.exists(), f"missing {json_path}"
    assert md_path.exists(), f"missing {md_path}"

    payload = json.loads(json_path.read_text())
    assert "overall" in payload
    assert "per_tercile" in payload
    overall = payload["overall"]
    for key in ("n_windows", "n_positive", "positive_rate", "density_q1", "density_q2"):
        assert key in overall, f"missing overall.{key}"
    per = payload["per_tercile"]
    assert isinstance(per, list)
    assert len(per) == 3
    names = {r["tercile"] for r in per}
    assert names == {"low", "medium", "high"}


def test_cli_missing_homography_exits_nonzero(tmp_path, fixture_paths):
    tracks, graphs, _ = fixture_paths
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "density_diagnostic.py"),
        "--tracks",
        str(tracks),
        "--graphs_dir",
        str(graphs),
        "--output_dir",
        str(tmp_path / "nope"),
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert "homography" in (r.stderr + r.stdout).lower()
