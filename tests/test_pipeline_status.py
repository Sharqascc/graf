from pathlib import Path

import pytest

from graf.utils.pipeline_status import (
    count_real_files,
    print_pipeline_status,
    should_skip,
    stage_status_icon,
)


def test_should_skip_ignored_dirs():
    assert should_skip(Path(".git/config"))
    assert should_skip(Path("__pycache__/module.pyc"))
    assert should_skip(Path("venv/lib/site.py"))
    assert not should_skip(Path("src/graf/__init__.py"))


def test_should_skip_ignored_files():
    assert should_skip(Path(".DS_Store"))
    assert not should_skip(Path("README.md"))


def test_count_real_files_empty_dir(tmp_path):
    assert count_real_files(tmp_path) == 0


def test_count_real_files_counts_only_real_files(tmp_path):
    (tmp_path / "real.txt").write_text("hello")
    (tmp_path / ".gitkeep").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "data.json").write_text("{}")
    assert count_real_files(tmp_path) == 2


def test_stage_status_icon():
    assert stage_status_icon(False, 0) == "MISSING"
    assert stage_status_icon(True, 0) == "SCAFFOLD_ONLY"
    assert stage_status_icon(True, 1) == "ACTIVE"


def test_print_pipeline_status_nonexistent_root():
    with pytest.raises(FileNotFoundError):
        print_pipeline_status(Path("/nonexistent/path"), depth=1)


def test_print_pipeline_status_existing_root(capsys):
    root = Path.cwd()
    print_pipeline_status(root, depth=1)
    captured = capsys.readouterr()
    assert "GRAF RESEARCH PIPELINE DASHBOARD" in captured.out
    assert "PIPELINE STAGES" in captured.out
    assert "CURRENT TREE" in captured.out
