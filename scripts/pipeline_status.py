from pathlib import Path
from collections import defaultdict
import subprocess
import argparse

IGNORE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv"
}
IGNORE_FILES = {".DS_Store"}

PIPELINE_STAGES = [
    {"name": "1. Repository base", "paths": ["README.md", "pyproject.toml", "requirements", ".gitignore"]},
    {"name": "2. Configuration layer", "paths": ["configs/dataset", "configs/detection", "configs/tracking", "configs/homography", "configs/graph", "configs/ssm", "configs/model", "configs/train", "configs/eval"]},
    {"name": "3. Raw data ingestion", "paths": ["data/raw/videos", "data/raw/metadata", "data/raw/site_notes"]},
    {"name": "4. Interim processing", "paths": ["data/interim/frames", "data/interim/detections", "data/interim/tracks", "data/interim/homography", "data/interim/trajectories"]},
    {"name": "5. Processed research artifacts", "paths": ["data/processed/actor_trajectories", "data/processed/graphs", "data/processed/ssm_events", "data/processed/windows", "data/processed/labels"]},
    {"name": "6. Source modules", "paths": ["src/graf/data", "src/graf/detection", "src/graf/tracking", "src/graf/calibration", "src/graf/trajectories", "src/graf/ssm", "src/graf/graph", "src/graf/models", "src/graf/training", "src/graf/evaluation", "src/graf/visualization", "src/graf/utils"]},
    {"name": "7. Entry scripts", "paths": ["scripts/extract_frames.py", "scripts/run_detection.py", "scripts/run_tracking.py", "scripts/estimate_homography.py", "scripts/build_trajectories.py", "scripts/compute_ssm.py", "scripts/build_graphs.py", "scripts/make_windows.py", "scripts/train_model.py", "scripts/evaluate_model.py", "scripts/export_paper_results.py"]},
    {"name": "8. Studies and experiments", "paths": ["studies", "tests", "notebooks", "docs"]},
    {"name": "9. Outputs", "paths": ["outputs/qc", "outputs/figures", "outputs/tables", "outputs/models", "outputs/predictions", "outputs/logs"]},
]

def should_skip(path: Path):
    return any(part in IGNORE_DIRS for part in path.parts) or path.name in IGNORE_FILES

def count_real_files(path: Path):
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    count = 0
    for p in path.rglob("*"):
        if should_skip(p):
            continue
        if p.is_file() and p.name != ".gitkeep":
            count += 1
    return count

def get_git_info(root: Path):
    try:
        branch = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except:
        branch = "N/A"
    try:
        short_status = subprocess.check_output(["git", "-C", str(root), "status", "--short"], stderr=subprocess.DEVNULL).decode().strip()
    except:
        short_status = ""
    try:
        status_sb = subprocess.check_output(["git", "-C", str(root), "status", "-sb"], stderr=subprocess.DEVNULL).decode().strip()
    except:
        status_sb = ""
    clean = "clean" if short_status == "" else "has changes"
    sync = "unpushed or diverged" if ("ahead" in status_sb or "behind" in status_sb) else "synced/unknown"
    return branch, clean, sync, status_sb

def print_tree(directory: Path, max_depth: int, prefix: str = "", depth: int = 0):
    if depth > max_depth:
        return
    entries = [p for p in sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower())) if not should_skip(p)]
    total = len(entries)
    for i, entry in enumerate(entries):
        connector = "└── " if i == total - 1 else "├── "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            ext = "    " if i == total - 1 else "│   "
            print_tree(entry, max_depth, prefix + ext, depth + 1)

def stage_status_icon(all_exist, real_files):
    if not all_exist:
        return "MISSING"
    if real_files == 0:
        return "SCAFFOLD_ONLY"
    return "ACTIVE"

def main():
    parser = argparse.ArgumentParser(description="Show current GRAF pipeline status")
    parser.add_argument("--root", type=str, default=".", help="Repository root")
    parser.add_argument("--depth", type=int, default=4, help="Tree print depth")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository not found: {root}")

    branch, clean, sync, status_sb = get_git_info(root)

    print("=" * 100)
    print("GRAF RESEARCH PIPELINE DASHBOARD")
    print("=" * 100)
    print(f"Repo root         : {root}")
    print(f"Git branch        : {branch}")
    print(f"Working tree      : {clean}")
    print(f"Remote state      : {sync}")
    print(f"Git summary       : {status_sb if status_sb else 'No git summary available'}")

    print("\\n" + "=" * 100)
    print("PIPELINE STAGES")
    print("=" * 100)

    for stage in PIPELINE_STAGES:
        statuses = [(root / p).exists() for p in stage["paths"]]
        real_files = sum(count_real_files(root / p) for p in stage["paths"])
        icon = stage_status_icon(all(statuses), real_files)
        print(f"\\n{stage['name']}  -->  {icon}")
        for rel_path, exists in zip(stage["paths"], statuses):
            p = root / rel_path
            kind = "file" if p.is_file() else "dir"
            extra = f"{count_real_files(p)} real files" if p.exists() and p.is_dir() else ""
            print(f"  [{'OK' if exists else '--'}] {rel_path:<45} {kind:<4} {extra}")

    print("\\n" + "=" * 100)
    print("CURRENT TREE")
    print("=" * 100)
    print(root.name)
    print_tree(root, args.depth)

if __name__ == "__main__":
    main()
