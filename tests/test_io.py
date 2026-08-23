import json

from graf.utils.io import (
    ensure_dir,
    get_git_commit,
    write_json,
    write_jsonl,
    write_text,
)


def test_ensure_dir_creates(tmp_path):
    d = tmp_path / "sub" / "dir"
    result = ensure_dir(d)
    assert result.is_dir()

def test_write_json(tmp_path):
    p = tmp_path / "data.json"
    write_json(p, {"a": 1, "b": [2, 3]})
    loaded = json.loads(p.read_text())
    assert loaded == {"a": 1, "b": [2, 3]}

def test_write_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    rows = [{"x": 1}, {"x": 2}]
    write_jsonl(p, rows)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"x": 1}

def test_write_text(tmp_path):
    p = tmp_path / "note.txt"
    write_text(p, "hello")
    assert p.read_text() == "hello"

def test_get_git_commit_unknown(tmp_path):
    assert get_git_commit(tmp_path) == "unknown"
