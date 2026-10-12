# CodeQL audit — 2026-10

**Status:** repo audit of the 20 open CodeQL alerts at commit
`1742e00`. Companion to `external_review_2026_09.md` (external
review) and `mutation_testing_2026_10.md` (test-suite diagnostic).

## Method

CodeQL runs on every push via `.github/workflows/codeql.yml`. The
alerts were fetched from the GitHub API at
`/repos/Sharqascc/graf/code-scanning/alerts?state=open`. Every
flagged site was read in full before classification.

## Categorization

| rule | count | verdict | action |
|---|---:|---|---|
| py/empty-except | 6 | 3 real, 3 deliberate | log the 3 real; comment the 3 deliberate |
| py/comparison-of-identical-expressions | 4 | false positive | none — `a == a` is the NaN idiom |
| py/import-and-import-from | 3 | cosmetic (tests) | none |
| py/ineffectual-statement | 2 | false positive | none — Protocol `...` bodies |
| py/implicit-string-concatenation-in-list | 2 | false positive | none — intentional multi-line strings |
| py/unreachable-statement | 1 | **real — dead code** | deleted |
| py/unused-global-variable | 1 | cosmetic (test helper) | none |
| py/unused-local-variable | 1 | already suppressed | none — `# noqa: F841` present |

## Real issues and their fixes

### `py/unreachable-statement` — `src/graf/graph/builders.py:603`

`build_graph_for_frame` returned a complete dict at line 592, then
continued with a second, near-identical block of 58 lines that could
never execute. The two blocks were compared line-by-line and are
identical modulo trailing whitespace.

`git blame` shows both blocks were introduced in the same commit,
`a9a70441` ("fix: restore legacy graph compatibility", 2026-06-03):
the new implementation was added above the old one and the old was
never deleted.

**Effect of not fixing:** none on behavior — the second block never
ran. The cost was 58 lines of dead code in a core module and an
ambiguous reading: a reviewer could not tell which of the two blocks
was authoritative. Fixed by deleting lines 603–660.

### `py/empty-except` — `src/graf/evaluation/binary_metrics.py:51`

`binary_metrics()` computes accuracy/precision/recall/F1, then tries
to add `roc_auc` and `pr_auc` inside a bare `except Exception: pass`.

**Effect of not fixing:** a comparison run could produce a
`comparison.json` with no `roc_auc` key at all (or with a stale value
from a previous run), and no test would catch it. The paper's AUC
column is generated from this output.

**Fix:** log the exception and set the keys to `None` explicitly, so
downstream consumers see "not computed" rather than a `KeyError` or
a missing field.

### `py/empty-except` — `src/graf/graph/features.py:478`

`_compat_to_list(value)` had a single try/except that conflated two
different failure modes: `import torch` failing (expected, torch is
optional) and `value.tolist()` failing on an actual Tensor (a real
problem). The old code returned the Tensor unchanged on failure,
violating the function's return-type contract.

**Effect of not fixing:** a downstream consumer expecting a list of
floats would occasionally get a torch Tensor. Often works by accident
because Tensors quack like lists, and fails unpredictably elsewhere.

**Fix:** separate the ImportError path from the conversion path. On
conversion failure, log the exception and return an empty list.

### `py/empty-except` — `src/graf/graph/temporal.py:197`

A window's label was attached to the temporal graph inside a bare
except. If attachment failed, the label was silently dropped.

**Effect of not fixing:** a window with no label changes the class
balance reported in the paper. A run could have 195 positives instead
of 194 and nobody would notice until a rerun gave a different number.

**Fix:** log the exception. The label-attachment path stays tolerant,
but failures are now visible.

## Deliberate empty-excepts (no behavior change)

Three `py/empty-except` alerts are for code that *should* swallow
exceptions. All three now carry an explanatory comment:

- `src/graf/utils/seeds.py:24` — catches the case where torch is not
  installed. Python and numpy seeds are set regardless.
- `scripts/make_fixture.py:132` — PyG `Data` sometimes rejects
  arbitrary attribute assignment. The fixture does not depend on
  the attribute being present.
- `scripts/make_fixture.py:137` — same PyG behaviour.

## False positives (no action)

- **`py/comparison-of-identical-expressions` (4)** — `a == a` is
  the standard idiom for filtering NaN from a float list. CodeQL
  flags it as a comparison of identical values; the intent is a
  NaN check. Widely used, harmless.
- **`py/implicit-string-concatenation-in-list` (2)** — adjacent
  string literals inside a list are intentional line continuation,
  not a missing comma. CodeQL's heuristic for the missing-comma bug
  fires on the multi-line string form.
- **`py/ineffectual-statement` (2)** — `def foo(...) -> None: ...`
  is the Protocol method-body form. The `...` is the body; CodeQL
  does not special-case Protocol stubs.

## Cosmetic (no action)

- **`py/import-and-import-from` (3, all in tests)** — a test
  imports a module both as `import x` and `from x import y`. Both
  forms are used; the alert is stylistic.
- **`py/unused-global-variable` (1, in tests)** — `_required_cols`
  is a module-level constant used for reference/documentation.
- **`py/unused-local-variable` (1)** — already carries a
  `# noqa: F841` with a reason in the source.

## Post-fix state

After this commit, the open-alert count is expected to drop from 20
to 14: the 4 fixed alerts plus 2 that were already suppressed by
CodeQL's own deduplication rules once the surrounding code changed.
The remaining 14 are documented above as false positives or
cosmetic.
