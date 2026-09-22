"""Diagnostic mutation testing for src/graf/ssm/ttc.py.

Applies a hand-listed set of mutations (boundary flips, sign changes,
constant swaps), runs the relevant tests for each, and reports which
mutations the suite kills vs. lets survive. A survivor is a code change
that the tests do not constrain: the suite passes regardless.

This is a diagnostic artifact, not a gate. It is not run in CI. Rerun
manually after changes to ttc.py or its tests.

Usage:
    python scripts/mutate_ttc.py --out docs/mutation_testing_2026_10.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "src" / "graf" / "ssm" / "ttc.py"
TEST_FILES = [
    "tests/test_ssm.py",
    "tests/test_ssm_extended.py",
    "tests/test_metamorphic.py",
]

# (label, find_string, replace_string). find_string must appear exactly
# once in the file. Line numbers are computed from the current file at
# runtime; the table is ordered by appearance.
MUTATIONS: list[tuple[str, str, str]] = [
    # is_critical: 0 < ttc <= 3.0
    (
        "lower bound <= vs <",
        "0 < self.ttc_seconds <= 3.0",
        "0 <= self.ttc_seconds <= 3.0",
    ),
    (
        "upper bound < vs <=",
        "0 < self.ttc_seconds <= 3.0",
        "0 < self.ttc_seconds < 3.0",
    ),
    (
        "upper constant 3.0 -> 2.0",
        "0 < self.ttc_seconds <= 3.0",
        "0 < self.ttc_seconds <= 2.0",
    ),
    (
        "upper constant 3.0 -> 4.0",
        "0 < self.ttc_seconds <= 3.0",
        "0 < self.ttc_seconds <= 4.0",
    ),
    (
        "drop isfinite() in is_critical",
        "np.isfinite(self.ttc_seconds) and 0 < self.ttc_seconds <= 3.0",
        "0 < self.ttc_seconds <= 3.0",
    ),
    (
        "and -> or in is_critical",
        "np.isfinite(self.ttc_seconds) and 0 < self.ttc_seconds <= 3.0",
        "np.isfinite(self.ttc_seconds) or 0 < self.ttc_seconds <= 3.0",
    ),
    # severity
    ("severity <= 0 -> < 0", "if self.ttc_seconds <= 0:", "if self.ttc_seconds < 0:"),
    (
        "severity >= 5.0 -> > 5.0",
        "if self.ttc_seconds >= 5.0:",
        "if self.ttc_seconds > 5.0:",
    ),
    (
        "severity horizon 5.0 -> 4.0 (upper check)",
        "if self.ttc_seconds >= 5.0:",
        "if self.ttc_seconds >= 4.0:",
    ),
    (
        "severity divisor 5.0 -> 10.0",
        "return float(1.0 - (self.ttc_seconds / 5.0))",
        "return float(1.0 - (self.ttc_seconds / 10.0))",
    ),
    # already-in-collision check
    (
        "already-in-collision <= -> <",
        "if _d <= min_distance + _tol:",
        "if _d < min_distance + _tol:",
    ),
    (
        "already-in-collision returns nonzero ttc",
        'return TTCResult(0.0, None, True, "already_in_collision")',
        'return TTCResult(0.001, None, True, "already_in_collision")',
    ),
    # zero relative speed
    ("zero-speed < -> <=", "if rel_speed_sq < 1e-12:", "if rel_speed_sq <= 1e-12:"),
    (
        "zero-speed tolerance 1e-12 -> 1e-6",
        "if rel_speed_sq < 1e-12:",
        "if rel_speed_sq < 1e-6:",
    ),
    # noise floor
    (
        "closing-rate <= -> <",
        "if closing_rate <= noise_floor:",
        "if closing_rate < noise_floor:",
    ),
    (
        "noise-floor 1e-9 -> 1e-6",
        "1e-9 * float(np.linalg.norm(rel_pos)) * float(np.sqrt(rel_speed_sq)) + 1e-12",
        "1e-6 * float(np.linalg.norm(rel_pos)) * float(np.sqrt(rel_speed_sq)) + 1e-12",
    ),
    # quadratic coefficients
    (
        "b factor 2.0 -> 1.0",
        "b = 2.0 * float(np.dot(rel_pos, rel_vel))",
        "b = 1.0 * float(np.dot(rel_pos, rel_vel))",
    ),
    (
        "c sign - -> +",
        "c = float(np.dot(rel_pos, rel_pos) - min_distance**2)",
        "c = float(np.dot(rel_pos, rel_pos) + min_distance**2)",
    ),
    (
        "discriminant sign - -> +",
        "discriminant = b**2 - 4.0 * a * c",
        "discriminant = b**2 + 4.0 * a * c",
    ),
    (
        "discriminant factor 4.0 -> 2.0",
        "discriminant = b**2 - 4.0 * a * c",
        "discriminant = b**2 - 2.0 * a * c",
    ),
    # tangent-approach tolerance
    (
        "tangent check -_eps -> 0.0",
        "if discriminant < -_eps:",
        "if discriminant < 0.0:",
    ),
    (
        "closest_sep vector + -> -",
        "rel_pos + t_near * rel_vel",
        "rel_pos - t_near * rel_vel",
    ),
    # root selection
    (
        "neg root sign - -> +",
        "roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]",
        "roots = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]",
    ),
    (
        "positive-root t > 0 -> t >= 0",
        "positive_roots = [t for t in roots if t > 0]",
        "positive_roots = [t for t in roots if t >= 0]",
    ),
    (
        "min(root) -> max(root)",
        "ttc = float(min(positive_roots))",
        "ttc = float(max(positive_roots))",
    ),
]


def line_of(source: str, needle: str) -> int:
    for i, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return i
    return -1


def run_tests() -> bool:
    """Return True if the test suite passes (mutant survived)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *TEST_FILES],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    original = TARGET.read_text()

    # Sanity: baseline tests pass on unmutated source
    print("baseline (unmutated) test run...")
    if not run_tests():
        print("BASELINE FAILS — aborting. Fix tests first.")
        return 2
    print("baseline passes.\n")

    rows = []
    for label, find_s, replace_s in MUTATIONS:
        occurrences = original.count(find_s)
        if occurrences != 1:
            rows.append(
                (
                    label,
                    line_of(original, find_s),
                    "SKIPPED",
                    f"find string appears {occurrences}x (need exactly 1)",
                )
            )
            print(f"  SKIP  {label}: find string appears {occurrences}x")
            continue
        mutated = original.replace(find_s, replace_s, 1)
        try:
            TARGET.write_text(mutated)
            passed = run_tests()
            outcome = "survived" if passed else "killed"
        finally:
            TARGET.write_text(original)
        line = line_of(original, find_s)
        rows.append((label, line, outcome, ""))
        print(f"  {outcome:<8} {label}  (line {line})")

    # Restore (should already be) and confirm
    TARGET.write_text(original)

    # Report
    killed = sum(1 for _, _, o, _ in rows if o == "killed")
    survived = sum(1 for _, _, o, _ in rows if o == "survived")
    skipped = sum(1 for _, _, o, _ in rows if o == "SKIPPED")
    total = killed + survived
    rate = (killed / total * 100.0) if total else 0.0

    lines = [
        "# Mutation testing — `src/graf/ssm/ttc.py`",
        "",
        "**Diagnostic artifact, not a CI gate.** Rerun with",
        "`python scripts/mutate_ttc.py --out docs/mutation_testing_2026_10.md`",
        "after any change to `ttc.py` or its tests.",
        "",
        "## Method",
        "",
        "A hand-listed set of semantic mutations (boundary flips, sign changes,",
        "constant swaps) is applied one at a time to `ttc.py`. After each,",
        "the relevant tests run:",
        "",
        "```",
        *[f"  {f}" for f in TEST_FILES],
        "```",
        "",
        "A mutant is **killed** if the test suite fails on the mutated source,",
        "**survived** if the suite still passes. A survivor is a change the",
        "tests do not constrain.",
        "",
        "## Result",
        "",
        f"- **Killed:** {killed}/{total}",
        f"- **Survived:** {survived}/{total}",
        f"- **Survival rate:** {rate:.1f}%" if total else "",
        f"- Skipped (find string not unique): {skipped}",
        "",
        "| # | mutation | line | outcome |",
        "|---|----------|-----:|---------|",
    ]
    for i, (label, line, outcome, note) in enumerate(rows, 1):
        cell_note = f" — {note}" if note else ""
        lines.append(f"| {i} | {label} | {line} | {outcome}{cell_note} |")

    if rate >= 30:
        lines += [
            "",
            "## Reading",
            "",
            f"Survival rate is {rate:.1f}%, above the 30% threshold at which",
            "mutation testing starts to flag weak coverage. Surviving mutants",
            "identify specific code paths the tests do not constrain. Each",
            "survivor above is a candidate for a targeted test.",
        ]
    else:
        lines += [
            "",
            "## Reading",
            "",
            f"Survival rate is {rate:.1f}%, below the 30% threshold at which",
            "mutation testing starts to flag weak coverage. The suite",
            "constrains the ttc.py behavior in the sampled mutation set.",
        ]

    # Interpreting-the-survivors section is written as a static block;
    # reruns of this tool overwrite the file, so keep it here rather than
    # in a separate doc.
    interpretation = [
        "",
        "## Interpreting the survivors",
        "",
        "Some reported survivors are semantically equivalent mutations",
        "(the mutator over-counts these). See the current doc for the",
        "classification of survivors into equivalent / boundary-condition /",
        "behavioral, and for follow-up candidates.",
    ]
    lines += interpretation

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    print(f"killed {killed}/{total}, survived {survived}/{total} ({rate:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
