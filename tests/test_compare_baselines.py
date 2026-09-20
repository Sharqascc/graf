"""Tests for scripts/compare_baselines.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import compare_baselines as cb

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------


def test_make_model_returns_known_classes():
    from graf.models.baselines import (
        LogisticRegressionBaseline,
        MajorityClassBaseline,
        MLPBaseline,
        RandomForestBaseline,
    )

    assert isinstance(cb._make_model("majority", 0), MajorityClassBaseline)
    assert isinstance(cb._make_model("logreg", 0), LogisticRegressionBaseline)
    assert isinstance(cb._make_model("rf", 0), RandomForestBaseline)
    assert isinstance(cb._make_model("mlp", 0), MLPBaseline)


def test_make_model_unknown_raises():
    with pytest.raises(ValueError, match="unknown tabular model"):
        cb._make_model("nope", 0)


def test_make_model_gcn_raises_in_tabular_factory():
    # GCN is handled separately; the tabular factory should not accept it
    with pytest.raises(ValueError):
        cb._make_model("gcn", 0)


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------


from torch_geometric.data import Data as _PyGData


def _make_fake_window(seed: int) -> _PyGData:
    """Return a real PyG Data object with every attribute the temporal
    window builder (src/graf/graph/temporal.py) and the feature extractor
    (src/graf/models/baselines.py) inspect.

    Required by temporal.py:  x, edge_index, edge_attr, pos, track_ids
    Optional but used by extractor: num_nodes, frame_id, video_id,
        actor_class_index, node_frame_index
    """
    rng = np.random.default_rng(seed)
    data = _PyGData(
        x=torch.tensor(rng.standard_normal((3, 5)), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.tensor(rng.standard_normal((2, 4)), dtype=torch.float32),
        pos=torch.tensor(rng.standard_normal((3, 2)), dtype=torch.float32),
        y=torch.tensor([1.0], dtype=torch.float32),
        num_nodes=3,
    )
    data.frame_id = int(seed)
    data.video_id = "v"
    data.track_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    data.actor_class_index = torch.tensor([0, 0, 0], dtype=torch.long)
    data.node_frame_index = torch.tensor([0, 0, 0], dtype=torch.long)
    return data


class _FakeWindow:
    """Backwards-compat wrapper that returns a real PyG Data object.
    Kept so the tiny-synthetic builder below doesn't have to change."""

    def __init__(self, seed: int):
        self._d = _make_fake_window(seed)

    def __getattr__(self, name):
        return getattr(self._d, name)


# Alias for internal use
def _FakeWindow_fn(seed):
    return _make_fake_window(seed)


class _FakeDS:
    def __init__(self, n: int):
        self._windows = [_make_fake_window(i) for i in range(n)]

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, i):
        return self._windows[i]


def test_features_for_windows_shape():
    ds = _FakeDS(4)
    X = cb._features_for_windows(ds, [0, 1, 2, 3])
    assert X.ndim == 2
    assert X.shape[0] == 4
    assert X.shape[1] > 0  # some feature dimension


def test_features_for_windows_subset():
    ds = _FakeDS(10)
    X = cb._features_for_windows(ds, [2, 5, 7])
    assert X.shape[0] == 3


# ------------------------------------------------------------------
# End-to-end: smallest possible run
# ------------------------------------------------------------------


def _write_tiny_synthetic(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a minimal tracks.jsonl + homography + graphs dir.

    Two frames of three actors close enough to be 'conflicting' when
    the distance threshold is set small; produces a 3-window dataset.
    """
    tracks = tmp_path / "tracks.jsonl"
    # add_world_coords projects bbox bottom-centers, so we synthesize a
    # bbox whose bottom-center is exactly our intended (x_m, y_m) under an
    # identity homography. Bottom center = ((x1+x2)/2, y2).
    #
    # 30 frames so windowing (size=5, stride=2) yields (30-5)//2+1 = 13
    # windows — enough to run 2-fold CV in the test below.
    N_FRAMES = 60
    # Design: actor 2 approaches actor 1 from 15 m away at 0.5 m/frame.
    # Distance crosses the 5 m conflict threshold around frame 20, so
    # early windows are label-negative and late windows are
    # label-positive. Actor 3 sits at x=100 and never conflicts. Mixed
    # labels are needed for logistic regression; 60 frames give ~28
    # windows so MLPBaseline's internal train/val split also has enough
    # samples per class.
    with tracks.open("w") as f:
        for frame in range(N_FRAMES):
            positions = [
                (1, 0.0),
                (2, max(15.0 - frame * 0.5, -5.0)),
                (3, 100.0),
            ]
            for tid, x_m in positions:
                y_m = 0.0
                bx1 = x_m - 5.0
                bx2 = x_m + 5.0
                by2 = y_m
                by1 = y_m - 10.0
                f.write(
                    json.dumps(
                        {
                            "video_id": "v",
                            "frame_idx": frame,
                            "track_id": tid,
                            "confidence": 0.9,
                            "bbox_xyxy": [bx1, by1, bx2, by2],
                            "x_m": x_m,
                            "y_m": y_m,
                            "vx_mps": 0.0,
                            "vy_mps": 0.0,
                        }
                    )
                    + "\n"
                )

    homography = tmp_path / "h.yaml"
    homography.write_text(
        "H:\n- [1.0, 0.0, 0.0]\n- [0.0, 1.0, 0.0]\n- [0.0, 0.0, 1.0]\n"
    )

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    for fi in range(N_FRAMES):
        torch.save(_make_fake_window(fi), graphs_dir / f"v_f{fi:06d}.pt")
    return tracks, homography, graphs_dir


def test_main_with_majority_only(tmp_path):
    tracks, homography, graphs_dir = _write_tiny_synthetic(tmp_path)
    out = tmp_path / "out"
    rc = cb.main(
        [
            "--tracks",
            str(tracks),
            "--graphs_dir",
            str(graphs_dir),
            "--homography_config",
            str(homography),
            "--output_dir",
            str(out),
            "--models",
            "majority",
            "--num_folds",
            "2",
            "--label-source",
            "proximity",
            "--distance_threshold",
            "5.0",
            "--min_interaction_frames",
            "1",
            "--split",
            "random",
        ]
    )
    assert rc == 0
    payload = json.loads((out / "comparison.json").read_text())
    assert "results" in payload
    assert len(payload["results"]) == 1
    r = payload["results"][0]
    assert r["model"] == "majority"
    assert 0.0 <= r["mean_accuracy"] <= 1.0
    assert 0.0 <= r["pooled_accuracy"] <= 1.0


def test_main_with_all_tabular_models(tmp_path):
    tracks, homography, graphs_dir = _write_tiny_synthetic(tmp_path)
    out = tmp_path / "out"
    rc = cb.main(
        [
            "--tracks",
            str(tracks),
            "--graphs_dir",
            str(graphs_dir),
            "--homography_config",
            str(homography),
            "--output_dir",
            str(out),
            "--models",
            "majority,logreg,rf,mlp",
            "--num_folds",
            "2",
            "--label-source",
            "proximity",
            "--distance_threshold",
            "5.0",
            "--min_interaction_frames",
            "1",
            "--split",
            "random",
        ]
    )
    assert rc == 0
    payload = json.loads((out / "comparison.json").read_text())
    names = [r["model"] for r in payload["results"]]
    assert names == ["majority", "logreg", "rf", "mlp"]


def test_main_rejects_unknown_model(tmp_path):
    tracks, homography, graphs_dir = _write_tiny_synthetic(tmp_path)
    with pytest.raises(SystemExit, match="unknown model"):
        cb.main(
            [
                "--tracks",
                str(tracks),
                "--graphs_dir",
                str(graphs_dir),
                "--homography_config",
                str(homography),
                "--output_dir",
                str(tmp_path / "out"),
                "--models",
                "does-not-exist",
            ]
        )


def test_cli_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compare_baselines.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--tracks",
        "--graphs_dir",
        "--homography_config",
        "--models",
        "--num_folds",
        "--split",
        "--label-source",
        "--ttc-threshold-seconds",
    ):
        assert flag in r.stdout


# ------------------------------------------------------------------
# Fold summary helpers
# ------------------------------------------------------------------


def test_negative_count_basic():
    assert cb._negative_count(np.array([1, 1, 0, 0, 0])) == 3
    assert cb._negative_count(np.array([1, 1, 1])) == 0
    assert cb._negative_count([0, 0]) == 2


def test_fold_auc_summary_all_nan():
    s = cb._fold_auc_summary([float("nan"), float("nan")])
    assert s["n_valid_folds"] == 0
    assert s["mean_auc"] != s["mean_auc"]  # NaN
    assert s["wilcoxon_p_vs_0_5"] is None


def test_fold_auc_summary_basic():
    s = cb._fold_auc_summary([0.6, 0.7, 0.8, 0.9, 1.0])
    assert s["n_valid_folds"] == 5
    assert abs(s["mean_auc"] - 0.8) < 1e-9
    # With 5 folds, minimum two-sided Wilcoxon p is ~0.0625.
    assert s["wilcoxon_p_vs_0_5"] is not None
    assert s["wilcoxon_p_vs_0_5"] >= 0.05


def test_fold_auc_summary_mixed_around_half():
    s = cb._fold_auc_summary([0.5, 0.5, 0.5, 0.5, 0.5])
    # No variation -> wilcoxon undefined.
    assert s["wilcoxon_p_vs_0_5"] is None


def test_fold_auc_summary_fewer_than_five_folds():
    # n<5 -> no wilcoxon attempted.
    s = cb._fold_auc_summary([0.6, 0.7, 0.8])
    assert s["n_valid_folds"] == 3
    assert s["wilcoxon_p_vs_0_5"] is None


# ------------------------------------------------------------------
# Threshold calibration
# ------------------------------------------------------------------


def test_choose_threshold_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    t = cb._choose_threshold(scores, labels)
    assert 0.2 < t <= 0.8


def test_choose_threshold_all_same_class_returns_half():
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([1, 1, 1])
    assert cb._choose_threshold(scores, labels) == 0.5
    labels0 = np.array([0, 0, 0])
    assert cb._choose_threshold(scores, labels0) == 0.5


def test_choose_threshold_single_unique_score_returns_half():
    scores = np.array([0.3, 0.3, 0.3])
    labels = np.array([0, 1, 1])
    assert cb._choose_threshold(scores, labels) == 0.5


def test_choose_threshold_inverted_scores_picks_boundary():
    # Positives have LOW scores. Youden's J still finds a threshold that
    # separates them; the returned threshold will sit near the low end.
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1, 1, 0, 0])
    t = cb._choose_threshold(scores, labels)
    # Best J: predict >= t as positive. t near 0.8 predicts only the two
    # high scorers (which are negatives) -> bad. t near 0.1 predicts all
    # positives. Verify the returned t is not blindly 0.5.
    assert t != 0.5
