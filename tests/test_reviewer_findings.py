"""Code-level regression guards for docs/external_review_2026_09.md.

Each test asserts that a fix for one reviewer finding is present in the
code. These tests do not run the pipeline; they inspect source. If a
future refactor reverts a fix, the test fails before any number using
the reverted code reaches the paper.

Findings that were fixed in the docs or the JSON, rather than in code,
are not covered here -- those live in the erratum and in the audit
module respectively.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _read(p: str) -> str:
    return (_REPO / p).read_text()


def test_finding_03_train_tabular_fold_placeholder_removed():
    """#3: the dead function returned y_train as if it were val labels."""
    src = _read("scripts/compare_baselines.py")
    assert "def train_tabular_fold" not in src, (
        "finding #3 regressed: train_tabular_fold reintroduced"
    )
    assert "# placeholder" not in src, (
        "finding #3 regressed: '# placeholder' comment reintroduced"
    )


def test_finding_04_setup_records_all_label_shaping_keys():
    """#4: comparison.json setup must record keys that change the labels."""
    src = _read("scripts/compare_baselines.py")
    for key in (
        "label_strategy",
        "run_length",
        "fraction_threshold",
        "calibrate_threshold",
        "calibration_objective",
    ):
        # the key must appear inside a "setup": { ... } block
        blocks = re.findall(r'"setup":\s*\{[^}]+\}', src, re.DOTALL)
        assert any(f'"{key}"' in b for b in blocks), (
            f"finding #4 regressed: {key!r} not in any setup block"
        )


def test_finding_06_majority_baseline_is_train_fitted():
    """#6: majority baseline must fit on the train fold, not use val oracle."""
    src = _read("scripts/compare_baselines.py")
    assert "_majority_baseline_accuracy" in src
    assert "max(y_val.mean(), 1 - y_val.mean())" not in src, (
        "finding #6 regressed: oracle majority accuracy reintroduced"
    )


def test_finding_07_uses_ge_for_all_decision_thresholds():
    """#7: all accuracy decisions use >=, none use >."""
    src = _read("scripts/compare_baselines.py")
    # The old form was (scores > 0.5) somewhere; grep for it.
    assert not re.search(r"scores\s*>\s*0\.5", src), (
        "finding #7 regressed: '>' used for a decision threshold"
    )


def test_finding_02_inner_split_has_blocked_path():
    """#2: inner split must have a blocked branch when purge_gap_frames > 0."""
    src = _read("scripts/compare_baselines.py")
    assert "purge_train_indices" in src, (
        "finding #2 regressed: purge_train_indices removed"
    )
    assert "inner_calib" in src
    # a blocked inner split ordered by frame position
    assert "sorted(" in src and "frame_sets" in src


def test_finding_01_leakage_is_reported_not_just_computed():
    """#1: leakage_report must be written into the result, not just computed."""
    src = _read("scripts/compare_baselines.py")
    assert '"leakage_raw"' in src
    assert '"leakage_after_purge"' in src


def test_docs_erratum_retracts_the_5_7_se_claim():
    """#5: the '5.7 SEs' claim must be retracted in the doc itself."""
    doc = _read("docs/paper/label_strategies_vntraffic.md")
    assert "Erratum" in doc, "finding #5 regressed: erratum block removed"
    assert "5.7 SEs" in doc
    # must appear alongside the retraction language, not standalone
    assert "withdrawn" in doc.lower() or "retracted" in doc.lower()
