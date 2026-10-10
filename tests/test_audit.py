"""Tests for graf.audit — the reviewer-checklist reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graf.audit import FAIL, PASS, WARN, audit_file, audit_payload, format_report


def _full_payload() -> dict:
    return {
        "setup": {
            "tracks": "t.jsonl",
            "graphs_dir": "g",
            "num_windows": 249,
            "label_source": "ttc",
            "label_strategy": "sustained",
            "run_length": 5,
            "ttc_threshold_seconds": 1.5,
            "ttc_distance_threshold": 3.0,
            "split": "blocked",
            "calibrate_threshold": True,
            "calibration_objective": "accuracy",
            "purge_gap_frames": 10,
        },
        "labels": {"majority_baseline": 0.779},
        "results": [
            {
                "model": "rf",
                "fold_accuracy": [0.7, 0.72],
                "fold_auc": [0.75, 0.78],
                "fold_majority": [0.77, 0.79],
                "fold_thresholds": [0.4, 0.5],
                "leakage_after_purge": {"leaked": 0, "total": 249},
            }
        ],
    }


def _by_id(checks):
    return {c.id: c for c in checks}


def test_all_checks_pass_on_full_payload():
    checks = audit_payload(_full_payload())
    assert all(c.status == PASS for c in checks), [
        (c.id, c.status, c.detail) for c in checks if c.status != PASS
    ]


def test_missing_setup_key_fails():
    p = _full_payload()
    del p["setup"]["run_length"]
    checks = _by_id(audit_payload(p))
    assert checks["#4"].status == FAIL
    assert "run_length" in checks["#4"].detail


def test_missing_purge_gap_fails():
    p = _full_payload()
    del p["setup"]["purge_gap_frames"]
    checks = _by_id(audit_payload(p))
    assert checks["#1"].status == FAIL


def test_zero_purge_gap_warns():
    p = _full_payload()
    p["setup"]["purge_gap_frames"] = 0
    checks = _by_id(audit_payload(p))
    assert checks["#1"].status == WARN


def test_nonzero_leakage_warns():
    p = _full_payload()
    p["results"][0]["leakage_after_purge"] = {"leaked": 3, "total": 249}
    checks = _by_id(audit_payload(p))
    assert checks["#1b"].status == WARN


def test_calibration_pairing_requires_thresholds():
    p = _full_payload()
    p["results"][0]["fold_thresholds"] = []
    checks = _by_id(audit_payload(p))
    assert checks["#2"].status == WARN


def test_calibration_disabled_marks_pass():
    p = _full_payload()
    p["setup"]["calibrate_threshold"] = False
    p["results"][0]["fold_thresholds"] = []
    checks = _by_id(audit_payload(p))
    assert checks["#2"].status == PASS


def test_bad_calibration_objective_fails():
    p = _full_payload()
    p["setup"]["calibration_objective"] = "f1"
    checks = _by_id(audit_payload(p))
    assert checks["#2"].status == FAIL


def test_missing_per_fold_data_warns():
    p = _full_payload()
    del p["results"][0]["fold_accuracy"]
    checks = _by_id(audit_payload(p))
    assert checks["#7"].status == WARN


def test_missing_results_warns_not_fails():
    p = _full_payload()
    p["results"] = []
    checks = _by_id(audit_payload(p))
    assert checks["#1b"].status == WARN
    assert checks["#7"].status == WARN


def test_audit_file_returns_exit_1_on_fail(tmp_path):
    p = tmp_path / "c.json"
    bad = _full_payload()
    del bad["setup"]["label_strategy"]
    p.write_text(json.dumps(bad))
    checks, code = audit_file(p)
    assert code == 1
    assert any(c.status == FAIL for c in checks)


def test_audit_file_returns_exit_0_on_pass(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_full_payload()))
    checks, code = audit_file(p)
    assert code == 0


def test_audit_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_file(tmp_path / "nope.json")


def test_format_report_contains_each_check_id():
    checks = audit_payload(_full_payload())
    report = format_report(checks, "c.json")
    for c in checks:
        assert c.id in report
    assert "warning(s)" in report
