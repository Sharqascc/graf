from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_git_commit(repo_dir: str | Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def write_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def snapshot_environment(repo_dir: str | Path, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    commit = get_git_commit(repo_dir)
    write_text(out_dir / "git_commit.txt", commit)

    env = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }

    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch"] = None

    try:
        import torch_geometric
        env["torch_geometric"] = torch_geometric.__version__
    except Exception:
        env["torch_geometric"] = None

    write_json(out_dir / "environment.json", env)
