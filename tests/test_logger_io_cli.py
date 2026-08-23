import logging

import pytest

from graf.cli import build_parser, main
from graf.utils.io import snapshot_environment
from graf.utils.logger import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO

def test_get_logger_respects_env(monkeypatch):
    monkeypatch.setenv("GRAF_LOG_LEVEL", "DEBUG")
    logger = get_logger("test_logger_env")
    assert logger.level == logging.DEBUG

def test_snapshot_environment_creates_files(tmp_path):
    snapshot_environment(".", tmp_path)
    assert (tmp_path / "git_commit.txt").exists()
    assert (tmp_path / "environment.json").exists()
    import json
    env = json.loads((tmp_path / "environment.json").read_text())
    assert "timestamp_utc" in env
    assert "python" in env

def test_build_parser_has_commands():
    parser = build_parser()
    assert any(action.dest == "command" for action in parser._actions)

def test_main_without_args(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["graf"])
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "GRAF: graph-based surrogate safety analysis pipeline" in captured.out

def test_main_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["graf", "--help"])
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
