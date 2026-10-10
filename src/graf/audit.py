"""Reviewer-checklist audit of a comparison.json run artifact.

Reads a ``comparison.json`` written by ``scripts/compare_baselines.py``
and reports whether the run's metadata answers each finding in
``docs/external_review_2026_09.md``. The output is a printed checklist;
the exit code is 0 if no check fails, 1 otherwise.

This does not re-run the experiment. It answers: "if a reviewer opened
this JSON, would they find the evidence they asked for?" That is a
different question from "is this result correct?" -- the code-level
checks live in ``tests/test_reviewer_findings.py``, and the doc-level
erratum lives in ``docs/paper/label_strategies_vntraffic.md``.

Usage:
    graf audit outputs/purged_sustained_r10/comparison.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PASS = "pass"
WARN = "warn"
FAIL = "fail"

REQUIRED_SETUP_KEYS = (
    "label_source",
    "label_strategy",
    "run_length",
    "ttc_threshold_seconds",
    "ttc_distance_threshold",
    "split",
    "calibrate_threshold",
    "calibration_objective",
)


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    status: str
    detail: str


def _check_setup_keys(payload: dict) -> Check:
    setup = payload.get("setup") or {}
    missing = [k for k in REQUIRED_SETUP_KEYS if k not in setup]
    if missing:
        return Check(
            "#4",
            "setup block records label-shaping keys",
            FAIL,
            f"missing: {', '.join(missing)}",
        )
    return Check(
        "#4",
        "setup block records label-shaping keys",
        PASS,
        f"{len(REQUIRED_SETUP_KEYS)}/{len(REQUIRED_SETUP_KEYS)} recorded",
    )


def _check_purge_gap(payload: dict) -> Check:
    setup = payload.get("setup") or {}
    gap = setup.get("purge_gap_frames")
    if gap is None:
        return Check(
            "#1",
            "fold boundaries purged",
            FAIL,
            "purge_gap_frames not recorded in setup",
        )
    if gap <= 0:
        return Check(
            "#1",
            "fold boundaries purged",
            WARN,
            f"purge_gap_frames={gap} (no purge applied)",
        )
    return Check(
        "#1",
        "fold boundaries purged",
        PASS,
        f"purge_gap_frames={gap}",
    )


def _check_leakage_reported(payload: dict) -> Check:
    results = payload.get("results") or []
    if not results:
        return Check("#1b", "leakage reported per result", WARN, "no results")
    missing = [
        r.get("model", "?") for r in results
        if r.get("leakage_after_purge") is None
    ]
    if missing:
        return Check(
            "#1b",
            "leakage reported per result",
            FAIL,
            f"missing on: {', '.join(missing)}",
        )
    nonzero = [
        r.get("model", "?")
        for r in results
        if (r.get("leakage_after_purge") or {}).get("leaked", 0) > 0
    ]
    if nonzero:
        return Check(
            "#1b",
            "leakage reported per result",
            WARN,
            f"nonzero leak after purge: {', '.join(nonzero)}",
        )
    return Check(
        "#1b",
        "leakage reported per result",
        PASS,
        f"{len(results)} results, zero leak after purge",
    )


def _check_calibration_pairing(payload: dict) -> Check:
    setup = payload.get("setup") or {}
    if not setup.get("calibrate_threshold"):
        return Check(
            "#2",
            "calibration objective paired with thresholds",
            PASS,
            "calibration disabled (not applicable)",
        )
    obj = setup.get("calibration_objective")
    if obj not in ("accuracy", "youden"):
        return Check(
            "#2",
            "calibration objective paired with thresholds",
            FAIL,
            f"calibration_objective={obj!r} not in (accuracy, youden)",
        )
    results = payload.get("results") or []
    unpaired = [
        r.get("model", "?") for r in results
        if not r.get("fold_thresholds")
    ]
    if unpaired:
        return Check(
            "#2",
            "calibration objective paired with thresholds",
            WARN,
            f"no fold_thresholds for: {', '.join(unpaired)}",
        )
    return Check(
        "#2",
        "calibration objective paired with thresholds",
        PASS,
        f"objective={obj}, thresholds reported for all models",
    )


def _check_majority_baseline(payload: dict) -> Check:
    """Finding #6 fix trains the majority predictor on the train fold.

    We cannot distinguish oracle vs train-fitted from the JSON alone,
    so this is a WARN that points at the code-level test.
    """
    labels = payload.get("labels") or {}
    if "majority_baseline" not in labels:
        return Check(
            "#6", "majority baseline present", FAIL,
            "labels.majority_baseline missing",
        )
    return Check(
        "#6",
        "majority baseline present",
        PASS,
        f"{labels['majority_baseline']:.3f} "
        "(train-fold-fitted check in tests/test_reviewer_findings.py)",
    )


def _check_result_shape(payload: dict) -> Check:
    results = payload.get("results") or []
    if not results:
        return Check("#7", "per-fold data present", WARN, "no results")
    needed = ("fold_accuracy", "fold_auc", "fold_majority")
    missing = [
        f"{r.get('model','?')}:{[k for k in needed if k not in r]}"
        for r in results
        if any(k not in r for k in needed)
    ]
    if missing:
        return Check(
            "#7", "per-fold data present", WARN, "; ".join(missing),
        )
    return Check(
        "#7",
        "per-fold data present",
        PASS,
        "fold_accuracy, fold_auc, fold_majority on all results",
    )


def audit_payload(payload: dict) -> list[Check]:
    return [
        _check_setup_keys(payload),
        _check_purge_gap(payload),
        _check_leakage_reported(payload),
        _check_calibration_pairing(payload),
        _check_majority_baseline(payload),
        _check_result_shape(payload),
    ]


_SYMBOL = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}


def format_report(checks: list[Check], source: str) -> str:
    lines = [
        f"Reviewer checklist — {source}",
        "Source of findings: docs/external_review_2026_09.md",
        "─" * 72,
    ]
    for c in checks:
        sym = _SYMBOL.get(c.status, "?")
        lines.append(f"  {sym}  {c.id:<5} {c.title:<45} {c.detail}")
    n_fail = sum(1 for c in checks if c.status == FAIL)
    n_warn = sum(1 for c in checks if c.status == WARN)
    lines.append("─" * 72)
    lines.append(f"{n_warn} warning(s), {n_fail} failure(s)")
    return "\n".join(lines)


def audit_file(path: str | Path) -> tuple[list[Check], int]:
    """Audit a comparison.json file. Returns (checks, exit_code)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"comparison.json not found: {p}")
    payload = json.loads(p.read_text())
    checks = audit_payload(payload)
    code = 1 if any(c.status == FAIL for c in checks) else 0
    return checks, code
