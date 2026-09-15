"""End-to-end test: synthetic tracks -> graphs -> windows -> GCN CV accuracy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scripts.make_synthetic_dataset import generate

pytest.importorskip("torch_geometric")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    return generate(root, num_frames=60)


def test_generator_layout(synthetic):
    assert synthetic["tracks"].exists()
    assert synthetic["homography"].exists()
    graphs = sorted(synthetic["graphs"].glob("*.pt"))
    assert len(graphs) == 60
    g = torch.load(graphs[0], weights_only=False)
    assert g.x.size(1) == 19
    assert g.edge_attr.size(1) == 15
    assert g.y.numel() == 1


def test_labels_are_mixed(synthetic):
    ys = [
        torch.load(p, weights_only=False).y.item()
        for p in sorted(synthetic["graphs"].glob("*.pt"))
    ]
    assert 0 < sum(ys) < len(ys), f"degenerate labels: {ys}"


def test_training_beats_chance(synthetic, tmp_path):
    out_dir = tmp_path / "cv_out"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "train_conflict_pairs.py"),
        "--tracks",
        str(synthetic["tracks"]),
        "--graphs_dir",
        str(synthetic["graphs"]),
        "--homography_config",
        str(synthetic["homography"]),
        "--output_dir",
        str(out_dir),
        "--distance_threshold",
        "5.0",
        "--min_interaction_frames",
        "3",
        "--epochs",
        "20",
        "--num_folds",
        "3",
        "--seed",
        "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    metrics = json.loads((out_dir / "metrics_cv.json").read_text())
    assert metrics["num_positive"] > 0
    assert metrics["num_negative"] > 0
    assert metrics["mean_val_accuracy"] > 0.65, metrics
