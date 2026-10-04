"""Build docs/paper/manuscript.md from the individual section files.

The paper's sections are maintained as separate markdown files under
docs/paper/. This script concatenates them into a single manuscript
in reading order, stripping the per-file Status preamble that is
for authors, not for readers.

Usage:
    python scripts/build_manuscript.py            # write manuscript.md
    python scripts/build_manuscript.py --check    # exit 1 if out of date

The --check mode is intended for CI: it fails if the committed
manuscript.md does not match what the current sources would produce,
which catches hand-edits made to the compiled file instead of to the
section source.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "docs" / "paper"
OUTPUT = PAPER / "manuscript.md"

# Reading order of the paper's sections, top to bottom.
SECTIONS = [
    "introduction.md",  # includes the abstract
    "related_work.md",
    "method.md",
    "results.md",
    "statistical_analysis.md",
    "discussion.md",
    "conclusion.md",
    "references.md",
]

TITLE = "The window-level surrogate safety label is an activity proxy"

FRONTMATTER = (
    f"# {TITLE}\n\n"
    "*[Draft manuscript. Title, authors, affiliations to be finalized.]*\n\n"
    "- **Code:** https://github.com/Sharqascc/graf\n"
    "- **Data:** VNTraffic & AICC22-Custom, Zenodo record 18195750\n"
    "- **Pre-registration:** `docs/preregistration_sustained_r5.md`\n"
    "- **Reproducibility:** `docs/reproducibility.md`\n"
)

BACKMATTER = (
    "## Figures\n\n"
    "| figure | caption |\n"
    "|---|---|\n"
    "| `figures/fig1_auc_vntraffic.pdf` | VNTraffic - mean AUC per model, 95% bootstrap CI. |\n"
    "| `figures/fig2_pre_post_purge.pdf` | Pre- vs post-purge AUC for RF and the single-feature cue. |\n"
    "| `figures/fig3_cross_clip.pdf` | Cross-clip comparison. Direction replicates; CIs overlap. |\n"
    "| `figures/fig4_gcn_collapse.pdf` | The GCN matches the majority baseline on accuracy exactly. |\n\n"
    "---\n\n"
    "## Reproducibility statement\n\n"
    "- Code: https://github.com/Sharqascc/graf\n"
    "- Data: VNTraffic and AICC22-Custom, Zenodo record 18195750\n"
    "- Pipeline: three commands from a fresh clone, documented in `docs/reproducibility.md`\n"
    "- Pre-registration: `docs/preregistration_sustained_r5.md`, committed before the rerun\n"
    "- All result JSONs: `docs/paper/*.json`\n"
)


def strip_status_preamble(text: str) -> str:
    """Remove the Status paragraph and the horizontal rule after it.

    Each section file begins with a top-level heading, then a
    Status note for authors, then ---. The build keeps the heading
    and everything after the horizontal rule; the status note is
    dropped.
    """
    pattern = re.compile(r"\n\*\*Status:\*\*.*?\n\s*\n---\s*\n", re.DOTALL)
    return pattern.sub("\n", text, count=1)


def build() -> str:
    parts: list[str] = [FRONTMATTER, ""]
    for name in SECTIONS:
        path = PAPER / name
        if not path.exists():
            raise FileNotFoundError(f"section missing: {path}")
        text = path.read_text().rstrip("\n")
        parts.append(strip_status_preamble(text))
        parts.append("")
        parts.append("---")
        parts.append("")
    # Drop trailing separator before backmatter
    while parts and parts[-1] in ("", "---"):
        parts.pop()
    parts.append("")
    parts.append(BACKMATTER)
    return "\n".join(parts).rstrip("\n") + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--check", action="store_true", help="exit 1 if manuscript.md is out of date"
    )
    args = p.parse_args(argv)

    built = build()
    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT} does not exist", file=sys.stderr)
            return 1
        current = OUTPUT.read_text()
        if current != built:
            print(f"{OUTPUT} is out of date with the section sources.", file=sys.stderr)
            print("Run: python scripts/build_manuscript.py", file=sys.stderr)
            return 1
        print(f"{OUTPUT} is up to date")
        return 0

    OUTPUT.write_text(built)
    print(f"wrote {OUTPUT} ({len(built)} bytes, {len(built.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
