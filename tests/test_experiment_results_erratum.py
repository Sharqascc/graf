"""Guard for the 87.5% retraction in docs/experiment_results.md.

The 87.5% number was produced from uncommitted local data before the
boundary-leak fix in PR #39. It has not been reproduced. This test
asserts that the retraction is present in the doc and in the README,
so that neither can silently drop the caveat and leave the number
looking current.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DOC = _REPO / "docs" / "experiment_results.md"
_README = _REPO / "README.md"


def test_erratum_present_in_experiment_results():
    text = _DOC.read_text()
    assert "Erratum" in text, "erratum block removed from experiment_results.md"
    assert "87.5" in text
    assert "has not been reproduced" in text


def test_erratum_cites_the_splitter_change():
    text = _DOC.read_text()
    assert ("PR #39" in text) or ("boundary-leak fix" in text), (
        "erratum must cite why the number is stale"
    )


def test_erratum_points_at_current_result():
    text = _DOC.read_text()
    assert "baselines_sustained_r5_post_purge.md" in text
    assert "docs/paper/results.md" in text


def test_readme_marks_number_retracted():
    text = _README.read_text()
    assert "87.5%" in text
    assert "retracted" in text.lower(), "README must mark the 87.5% number as retracted"
    assert "baselines_sustained_r5_post_purge.md" in text


def test_erratum_before_body():
    """The erratum must appear in the top 30 lines of the file."""
    lines = _DOC.read_text().splitlines()
    top = chr(10).join(lines[:30])
    assert "Erratum" in top, "erratum must appear near the top of the doc, not buried"
