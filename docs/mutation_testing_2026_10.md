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

- **Killed:** 11/25
- **Survived:** 14/25
- **Survival rate:** 44.0%
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
| 9 | severity horizon 5.0 -> 4.0 (upper check) | 25 | survived |
| 10 | severity divisor 5.0 -> 10.0 | 27 | killed |
| 11 | already-in-collision <= -> < | 54 | survived |
| 12 | already-in-collision returns nonzero ttc | 55 | killed |
| 13 | zero-speed < -> <= | 58 | survived |
| 14 | zero-speed tolerance 1e-12 -> 1e-6 | 58 | survived |
| 15 | closing-rate <= -> < | 69 | survived |
| 16 | noise-floor 1e-9 -> 1e-6 | 67 | survived |
| 17 | b factor 2.0 -> 1.0 | 73 | killed |
| 18 | c sign - -> + | 74 | killed |
| 19 | discriminant sign - -> + | 76 | killed |
| 20 | discriminant factor 4.0 -> 2.0 | 76 | killed |
| 21 | tangent check -_eps -> 0.0 | 84 | survived |
| 22 | closest_sep vector + -> - | 86 | survived |
| 23 | neg root sign - -> + | 97 | survived |
| 24 | positive-root t > 0 -> t >= 0 | 98 | survived |
| 25 | min(root) -> max(root) | 103 | survived |

## Reading

Survival rate is 44.0%, above the 30% threshold at which
mutation testing starts to flag weak coverage. Surviving mutants
identify specific code paths the tests do not constrain. Each
survivor above is a candidate for a targeted test.

## Interpreting the survivors

The 44% rate above counts every mutation, including three that are
semantically equivalent to the original code. Classifying the 14
survivors:

### Semantically equivalent (false survivors; the mutator over-counts)

- **`drop isfinite() in is_critical`** — `0 < nan <= 3.0` already
  evaluates to `False`, so the `np.isfinite` guard is redundant in this
  expression. Removing it does not change behavior.
- **`severity <= 0 -> < 0`** — at `ttc_seconds == 0`, the mutated path
  falls through to the arithmetic `1.0 - 0/5.0 == 1.0`. Same output.
- **`severity >= 5.0 -> > 5.0`** — at `ttc_seconds == 5.0`, the mutated
  path falls through to `1.0 - 5.0/5.0 == 0.0`. Same output.

Real survival rate excluding these: **11/22 (50%)** — higher than the
raw 44%, because all three false survivors were counted in the
survivor column. The raw rate understates the gap in coverage.

### Boundary-condition survivors (real, narrow)

Six mutations flip a comparison operator on an exact boundary
(`already_in_collision`, zero relative speed, closing-rate noise floor,
positive-root filter). Tests exercise the neighborhoods but not the
exact equality cases.

### Behavioral survivors (real, material)

Four mutations change behavior on a range, not a point. Listed in
order of what they would mean for a paper claim:

1. **`min(root) -> max(root)` (line 103)** — the function could return
   the *exit* root (when actors leave the collision radius) instead of
   the *entry* root (time to first contact) and no test catches it.
   This is the TTC definition itself.
2. **`severity horizon 5.0 -> 4.0` (line 25)** — the severity scale
   collapses to 0 at ttc=4.5 in the mutated version; no test asserts
   severity on the (4.0, 5.0) interval.
3. **`tangent check -_eps -> 0.0` (line 84)** — removes the scale-aware
   tangent tolerance that the inline comment says exists to make
   rotation invariance deterministic. `test_metamorphic` did not catch
   it, so either the tolerance is not load-bearing on the tested inputs
   or the metamorphic test does not exercise near-tangent pairs.
4. **`noise-floor 1e-9 -> 1e-6` (line 67)** — widens the diverging/parallel
   rejection band by three orders of magnitude; not caught.

## Follow-up

Targeted tests for the four behavioral survivors are a candidate for a
later PR. The two that most affect the paper's results are the root
selection (`min` vs `max`) and the severity horizon; both are functions
the label generation depends on directly.
