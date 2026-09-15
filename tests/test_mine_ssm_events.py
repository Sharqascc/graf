"""Tests for scripts/mine_ssm_events.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import mine_ssm_events as script_mod

REPO = Path(__file__).resolve().parents[1]


def _values_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _values_json(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


_SAMPLE = [
    {
        "video_id": "v",
        "frame_idx": 0,
        "track_id_a": "a",
        "track_id_b": "b",
        "metric_name": "TTC",
        "value": 3.0,
    },
    {
        "video_id": "v",
        "frame_idx": 1,
        "track_id_a": "a",
        "track_id_b": "b",
        "metric_name": "TTC",
        "value": 1.0,
    },
    {
        "video_id": "v",
        "frame_idx": 2,
        "track_id_a": "a",
        "track_id_b": "b",
        "metric_name": "TTC",
        "value": 0.8,
    },
    {
        "video_id": "v",
        "frame_idx": 3,
        "track_id_a": "a",
        "track_id_b": "b",
        "metric_name": "TTC",
        "value": 2.0,
    },
]


def test_load_ssm_values_jsonl(tmp_path):
    p = _values_jsonl(tmp_path / "v.jsonl", _SAMPLE)
    df = script_mod.load_ssm_values(p)
    assert len(df) == 4
    assert set(df.columns) >= {"video_id", "frame_idx", "metric_name", "value"}


def test_load_ssm_values_json_array(tmp_path):
    p = _values_json(tmp_path / "v.json", _SAMPLE)
    df = script_mod.load_ssm_values(p)
    assert len(df) == 4


def test_load_ssm_values_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        script_mod.load_ssm_values(tmp_path / "nope.json")


def test_load_ssm_values_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    df = script_mod.load_ssm_values(p)
    assert len(df) == 0


def test_resolve_thresholds_json_override():
    t = script_mod.resolve_thresholds("configs/ssm", '{"TTC": 1.5}')
    assert t == {"TTC": 1.5}


def test_resolve_thresholds_configs_default():
    t = script_mod.resolve_thresholds(str(REPO / "configs" / "ssm"), None)
    assert "TTC" in t and t["TTC"] == 3.0
    assert "PET" in t and t["PET"] == 1.5
    assert "DRAC" in t and t["DRAC"] == 3.35


def test_main_writes_events(tmp_path):
    values = _values_jsonl(tmp_path / "v.jsonl", _SAMPLE)
    outdir = tmp_path / "out"
    rc = script_mod.main(
        [
            "--values-path",
            str(values),
            "--outdir",
            str(outdir),
            "--thresholds-json",
            '{"TTC": 1.5}',
        ]
    )
    assert rc == 0
    events_path = outdir / "ssm_events.jsonl"
    summary_path = outdir / "summary.json"
    assert events_path.exists()
    assert summary_path.exists()

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["metric_name"] == "TTC"
    assert events[0]["start_frame"] == 1
    assert events[0]["end_frame"] == 2

    summary = json.loads(summary_path.read_text())
    assert summary["num_events"] == 1
    assert summary["thresholds"] == {"TTC": 1.5}


def test_main_no_thresholds_writes_empty(tmp_path):
    values = _values_jsonl(tmp_path / "v.jsonl", _SAMPLE)
    outdir = tmp_path / "out"
    rc = script_mod.main(
        [
            "--values-path",
            str(values),
            "--outdir",
            str(outdir),
            "--thresholds-json",
            "{}",
        ]
    )
    assert rc == 0
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["num_events"] == 0


def test_main_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        script_mod.main(
            [
                "--values-path",
                str(tmp_path / "missing.json"),
                "--outdir",
                str(tmp_path / "out"),
                "--thresholds-json",
                '{"TTC": 1.5}',
            ]
        )


def test_script_cli_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "mine_ssm_events.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--values-path",
        "--outdir",
        "--config-dir",
        "--thresholds-json",
        "--min-duration-frames",
        "--max-frame-gap",
    ):
        assert flag in r.stdout
