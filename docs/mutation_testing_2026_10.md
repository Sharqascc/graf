# Mutation testing — `src/graf/ssm/ttc.py`

**Diagnostic artifact, not a CI gate.** Rerun with
`python scripts/mutate_ttc.py --out docs/mutation_testing_2026_10.md`
after any change to `ttc.py` or its tests.

## Method

A hand-listed set of semantic mutations (boundary flips, sign changes,
constant swaps) is applied one at a time to `ttc.py`. After each,
the relevant tests run:

```
  tests/test_ssm.py
  tests/test_ssm_extended.py
  tests/test_metamorphic.py
```

A mutant is **killed** if the test suite fails on the mutated source,
**survived** if the suite still passes. A survivor is a change the
tests do not constrain.

## Result

- **Killed:** 13/25
- **Survived:** 12/25
- **Kill rate:** 52.0%
- **Survival rate:** 48.0%
- Skipped (find string not unique): 0

| # | mutation | line | outcome |
|---|----------|-----:|---------|
| 1 | lower bound <= vs < | 17 | killed |
| 2 | upper bound < vs <= | 17 | killed |
| 3 | upper constant 3.0 -> 2.0 | 17 | killed |
| 4 | upper constant 3.0 -> 4.0 | 17 | killed |
| 5 | drop isfinite() in is_critical | 17 | survived |
| 6 | and -> or in is_critical | 17 | killed |
| 7 | severity <= 0 -> < 0 | 23 | survived |
| 8 | severity >= 5.0 -> > 5.0 | 25 | survived |
| 9 | severity horizon 5.0 -> 4.0 (upper check) | 25 | killed |
| 10 | severity divisor 5.0 -> 10.0 | 27 | killed |
| 11 | already-in-collision <= -> < | 74 | survived |
| 12 | already-in-collision returns nonzero ttc | 75 | killed |
| 13 | zero-speed < -> <= | 78 | survived |
| 14 | zero-speed tolerance 1e-12 -> 1e-6 | 78 | survived |
| 15 | closing-rate <= -> < | 89 | survived |
| 16 | noise-floor 1e-9 -> 1e-6 | 87 | survived |
| 17 | b factor 2.0 -> 1.0 | 93 | killed |
| 18 | c sign - -> + | 94 | killed |
| 19 | discriminant sign - -> + | 96 | killed |
| 20 | discriminant factor 4.0 -> 2.0 | 96 | killed |
| 21 | tangent check -_eps -> 0.0 | 104 | survived |
| 22 | closest_sep vector + -> - | 106 | survived |
| 23 | neg root sign - -> + | 117 | survived |
| 24 | positive-root t > 0 -> t >= 0 | 118 | survived |
| 25 | min(root) -> max(root) | 123 | killed |

## Reading

Survival rate is 48.0%, above the 30% threshold at
which mutation testing starts to flag weak coverage. Surviving
mutants identify specific code paths the tests do not constrain.
Each survivor above is a candidate for a targeted test.

## Errata — rate label fix (PR #48)

PR #45's version of this doc reported "Survival rate: 44.0%" from the
same code that now says "Survival rate: 48.0%". The number was
computed correctly but labeled backwards: `killed / total` is the
**kill** rate, not the survival rate. Both are now reported explicitly.
Historical survival for pre-PR-#48 code: 56.0%.

## Changes since PR #45

Four behavioral mutations that survived PR #45 now die:

- `min(root) -> max(root)` (line 123) — killed by
  `test_ttc_returns_entry_root_not_exit_root`, which asserts the
  function returns time to *first* contact, not time to separation.
- `severity horizon 5.0 -> 4.0` (line 25) — killed by
  `test_ttc_severity_at_horizon_interior`, which asserts severity on
  the (4.0, 5.0) interval.
- `already-in-collision returns nonzero ttc` (line 75) — killed by the
  new sentinel path from `test_ttc_non_finite_inputs_return_sentinel`.

Survival rate: 56.0% (PR #45 code) → 48.0% (PR #48 code).

## Interpreting the survivors

Classifying the 12 surviving mutations:

### Semantically equivalent (false survivors; the mutator over-counts)

- **`drop isfinite() in is_critical`** — `0 < nan <= 3.0` is `False`,
  so the `np.isfinite` guard is redundant. Removing it does not change
  behavior.
- **`severity <= 0 -> < 0`** — at `ttc_seconds == 0`, the mutated path
  falls through to `1.0 - 0/5.0 == 1.0`. Same output.
- **`severity >= 5.0 -> > 5.0`** — at `ttc_seconds == 5.0`, the mutated
  path falls through to `1.0 - 5.0/5.0 == 0.0`. Same output.

Three false survivors. Real survival excluding those: 9/22 (40.9%).

### Boundary-condition survivors (real, narrow)

Five mutations flip a comparison operator on an exact boundary
(`already_in_collision`, zero relative speed, closing-rate noise floor,
positive-root filter, negative-root sign). Tests exercise the
neighborhoods but not the exact equality cases. Adding tests here is
possible but low-value: the code paths are guard rails against float64
noise, not behavior the paper depends on.

### Behavioral survivors (real, worth a follow-up)

- **`zero-speed tolerance 1e-12 -> 1e-6`** (line 78) — widens the
  zero-relative-speed rejection band by six orders of magnitude. Not
  caught because no test constructs a near-zero-but-nonzero relative
  velocity.
- **`noise-floor 1e-9 -> 1e-6`** (line 87) — widens the
  diverging/parallel rejection band. Same shape.
- **`tangent check -_eps -> 0.0`** (line 104) — removes the scale-aware
  tangent tolerance the inline comment says exists for rotation
  invariance. `test_metamorphic` did not catch it.
- **`closest_sep vector + -> -`** (line 106) — inverts the closest-
  approach vector for the `no_collision_min_sep_*` status. Affects only
  the human-readable status string.
- **`neg root sign - -> +`** (line 117), **`positive-root t > 0 -> t >= 0`**
  (line 118) — root-selection boundary. The first is likely equivalent
  (swapping then min gives the same answer); the second is a t=0
  inclusion boundary no test hits.

The three that matter are the two tolerance widenings and the tangent
tolerance removal. All three are scaling constants chosen against
float64 behavior rather than physical quantities; tests would need
specific near-degenerate geometries. That is a follow-up, not a blocker.
