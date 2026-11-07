# TTC formula decision (item 16)

**Status:** pre-implementation audit. No code has been changed by this
document. Three separate TTC implementations exist in the repo, they
compute different physical quantities, and one of them generates every
label used in the paper. This document names the three, shows where
they diverge on a concrete input, and lays out three candidate label
definitions. **Item 16 is not resolved until one of those candidates is
chosen and the affected experiment is re-run.**

## Purpose

A reviewer who reads `src/graf/ssm/ttc.py`,
`src/graf/graph/edges.py`, and `scripts/label_strategies.py` will find
three different formulas that all carry the name "TTC." They are not
the same quantity. Two compute time-to-threshold; one computes
time-to-closest-approach. The paper reports numbers produced by the
third formula as if it were the first.

This document makes the divergence explicit and unblockable.

## The three implementations

### 1. SSM module — `compute_ttc_constant_velocity`

File: `src/graf/ssm/ttc.py`. The function solves the quadratic

    |rel_pos + t * rel_vel|^2 = collision_radius^2

for the smallest positive `t`. If no positive root exists, it returns
`inf` with a status that names the reason (`no_collision_min_sep_X`,
`diverging_or_parallel`, `zero_relative_speed`). If the pair is
already inside the radius at `t=0`, it returns `0.0` with status
`already_in_collision`.

| Quantity | Units |
|---|---|
| Inputs (positions, velocities) | m, m/s |
| `min_distance` | m |
| **Output (`ttc_seconds`)** | **s** |

Meaning: *time until separation reaches `min_distance` under constant
velocity, or inf if unreachable.*

### 2. Edge feature — `compute_ttc_simple`

File: `src/graf/graph/edges.py`. Same quadratic as #1, with two
differences:

- **Capped at `max_ttc = 10.0`**: cases that #1 would return as `inf`
  here return `10.0` instead.
- **No `already_in_collision` branch before PR #148.** That fix was
  merged (PR #148, `main` at `1535045412f6`). Before the fix, an
  already-inside-radius pair that was separating returned the *exit*
  root; after the fix, it returns `0.0`.

| Quantity | Units |
|---|---|
| Inputs | m, m/s |
| `collision_radius` | m |
| `max_ttc` | s |
| **Output (float)** | **s**, in `[0, max_ttc]` |

Meaning: *time until separation reaches `collision_radius`, capped at
`max_ttc` if no reachable collision.*

### 3. Label generator — `_frame_ttc_stats`

File: `scripts/label_strategies.py`. The per-frame statistic used by
every label strategy (`any`, `sustained`, `min_ttc`, `pair_fraction`,
`persistent_pair`). The formula:

    rel_pos       = pos_b - pos_a
    rel_vel       = vel_b - vel_a
    distance      = |rel_pos|
    closing_rate  = -dot(rel_pos, rel_vel)
    rss           = dot(rel_vel, rel_vel)
    ttc           = closing_rate / rss

with two gates:

    if distance >= distance_threshold:        continue  (skip)
    if rss < 1e-9 or closing_rate <= 0.5:     continue  (skip)

Only pairs that pass both gates contribute to the frame's criticality.
The frame is critical if **any** such pair has `ttc <= 1.5`.

| Quantity | Units |
|---|---|
| `distance_threshold` | m (3.0) |
| `closing_rate_threshold` | m/s (0.5) |
| `ttc_threshold_seconds` | s (1.5) |
| **Output (per-frame `min_ttc`, per-window label)** | **s / binary** |

## Dimensional analysis — is the label formula even in seconds?

Review 1 raised the possibility that the label formula is an
inverse-time quantity rather than a time. Dimensional check:

    rel_pos:       [m]
    rel_vel:       [m/s]
    closing_rate:  -dot(rel_pos, rel_vel)   ->  [m * m/s]  =  [m^2 / s]
    rss:           dot(rel_vel, rel_vel)    ->  [(m/s)^2]  =  [m^2 / s^2]
    ttc:           closing_rate / rss       ->  [m^2/s] / [m^2/s^2]  =  [s]

**The formula IS in seconds.** Review 1's second reading was correct.
But the seconds it computes are not the seconds computed by #1 and #2.

### The seconds it computes are TCPA, not TTC-to-threshold

For a pair with relative position `r` and relative velocity `v`:

- **TCPA** (time to closest approach) = `-dot(r, v) / dot(v, v)`. Time
  until the pair stops approaching. Independent of whether they hit.
- **TTC-to-threshold** = the `t` solving `|r + t*v| = radius`. Only
  finite when the pair actually reaches the radius.

**These two are equal only when the closest approach distance (DCPA) is
smaller than the threshold radius.** When DCPA > radius, TCPA is finite
but TTC-to-threshold is infinite. The label formula returns TCPA and
never checks DCPA.

## The counterexample

Two vehicles. Vehicle A stationary at `(0, 0)`. Vehicle B at `(2.8, 0)`
moving at `(-0.5, -2)` m/s.

    rel_pos       = (2.8, 0)               distance     = 2.8 m
    rel_vel       = (-0.5, -2)             |rel_vel|   = 2.06 m/s
    closing_rate  = -dot(rel_pos, rel_vel) = 1.4 m/s
    rss           = dot(rel_vel, rel_vel)  = 4.25 (m/s)^2
    ttc           = 1.4 / 4.25             = 0.329 s

### What the label generator sees

- `distance = 2.8 < distance_threshold = 3.0` → passes the gate
- `closing_rate = 1.4 > 0.5` → passes the second gate
- `ttc = 0.329 <= 1.5` → **frame is critical**

### What the edge feature sees

`compute_ttc_simple` runs the same pair through the quadratic with
`collision_radius = 1.5`:

    rx = 2.8, ry = 0, rvx = -0.5, rvy = -2
    a = 4.25, b = -2.8, c = 2.8^2 - 1.5^2 = 5.59
    discriminant = b^2 - 4ac = -87.19 < 0

`discriminant < 0` means the pair never reaches 1.5 m of separation.
The function returns `max_ttc = 10.0`.

### What the physics actually is

    DCPA = |rel_pos + tcpa * rel_vel|
         = |(2.8, 0) + 0.329 * (-0.5, -2)|
         = |(2.635, -0.658)|
         = 2.716 m

The pair passes **2.716 m apart**. They never touch (0 m), they never
enter the 1.5 m threshold. They are a clean non-conflict.

| Source | Says |
|---|---|
| Label | **critical** (TCPA = 0.329 s, within 3 m, closing) |
| Edge feature | **10.0 s** (`max_ttc`, i.e. no collision) |
| Physics | **they pass 2.716 m apart, never touch** |

**The model is being trained to flag a safe pass as a critical
conflict.**

## Why this makes the paper's current claim unsound

The paper's claim (per `docs/paper/results.md` and
`docs/paper/conclusion.md`) is that a graph model fails to learn the
label. That claim is about a label whose physical meaning is TCPA-gated
proximity, not TTC-to-threshold. Every feature and model in the
pipeline uses TTC-to-threshold in its input. The label uses a different
quantity. The model is not being asked to predict the physics that its
inputs carry.

The feature-label circularity documented in
`docs/audit/feature_label_circularity.md` (item 1) is real, but it is
not the whole picture. The features and the label are correlated at
the extremes (both fire when DCPA is near zero) but diverge in the
middle of the score range (label fires, feature does not, per the
counterexample above). That divergence is the exact region where
model accuracy is lost.

**No experiment should be re-run until a label definition is chosen
that measures the same quantity as the features.**

## Two decisions item 16 contains

### Decision A — vocabulary

Adopt one name per physical quantity:

| Name | Formula | Purpose |
|---|---|---|
| `ttc_to_threshold` | time until separation = collision_radius | feature on edges, canonical SSM |
| `tcpa` | `-dot(rel_pos, rel_vel) / rss` | diagnostic |
| `dcpa` | `|rel_pos + tcpa * rel_vel|` | diagnostic |
| `pet` | time between zone entries | separate SSM, zone-based |

Then `compute_ttc_simple` is documented as *TTC-to-threshold*, not
*TTC*. `_frame_ttc_stats` is documented as *TCPA-based*, not *TTC*.
No values change; only the names.

**Decision A has no research consequence.** It is purely a naming
clarity fix. It can be implemented in a small PR (rename function,
update docstrings, add deprecation alias). It does not change any
number in any result file.

### Decision B — what the label means

The label definition is a research choice. Three candidates, in
increasing effort:

#### B.0 — Keep the label formula, rename it honestly

Change only the paper and docstrings: "TCPA-based proximity label"
instead of "TTC label." The code stays as-is. The counterexample above
still exists — the model still learns to flag 2.7 m passes as
critical — but the paper no longer claims to predict TTC.

- Code change: **none**
- Paper change: replace "TTC label" everywhere with "TCPA-based
  proximity label"
- Physical soundness: **low** — the label still fires on non-events
- Rerun needed: **no** (results are what they are, just renamed)

#### B.1 — DCPA-gated TCPA

Change `_frame_ttc_stats` so a pair is critical only if:

    distance      < distance_threshold (3.0 m)     AND
    closing_rate  > 0.5 m/s                         AND
    tcpa          < ttc_threshold_seconds (1.5 s)   AND
    dcpa          < collision_radius (1.5 m)

The DCPA gate excludes near-miss passes. Per the counterexample, DCPA =
2.716 m > 1.5 m, so that pair would no longer be critical. The label
and the feature now agree on the pair not being a conflict.

- Code change: `_frame_ttc_stats` adds a DCPA computation and gate
  (~10 lines), plus a new config key for the DCPA threshold
- Paper change: "DCPA-gated TCPA label" — physically defensible
- Physical soundness: **medium-high** — fires only on pairs that would
  actually reach the radius
- Rerun needed: **yes** — every label, every `.pt`, every
  `comparison.json`, every paper number

Independence: the DCPA gate uses `rel_pos` and `rel_vel` directly;
`compute_ttc_simple` uses them to solve a quadratic. Same inputs,
different computation. No path from feature to label without
re-simulating the physics — the model cannot trivially read the label
off the features.

#### B.2 — PET-based label

Replace `_frame_ttc_stats` with a per-window PET computation using
`graf.ssm.pet.compute_pet_from_conflict_zone`. The window is critical
if the pair's PET (or any pair's PET) is below a threshold.

- Code change: larger — new per-window label generator, new
  `compute_pet_from_conflict_zone` call shape, new tests
- Prerequisite: each intersection needs a defined **conflict zone**
  (center + radius). `compute_pet_from_conflict_zone` takes these
  explicitly; nothing in the repo derives them from the intersection
  geometry. This is a genuine infrastructure gap.
- Paper change: "PET-based label" — physically standard for
  unsignalized intersections (FHWA treats PET as a primary SSM)
- Physical soundness: **high** — PET is the recommended measure for
  intersections where TTC is poorly defined
- Rerun needed: **yes**

Independence: PET is a zone-based measure, entirely independent of
edge TTC. No leakage possible.

## Recommendation for Decision B

**DCPA-gated TCPA (B.1)**, for three reasons:

1. **It closes the divergence without new infrastructure.**
   B.2 requires defining a conflict zone per intersection, which is a
   separate project. B.1 uses the same kinematic inputs the code
   already has.
2. **It preserves the paper's current comparison design.**
   Every experiment in `docs/paper/` asks "does the graph model beat
   the scalar?" That question survives. It's the label definition that
   changes, not the comparison.
3. **It fixes the specific counterexample.**
   The pair that passes 2.716 m apart no longer counts as critical.
   Every pair that legitimately reaches 1.5 m still does.

**B.2 (PET-based)** is a better paper in the long run. It aligns with
how the SSM literature actually treats intersections. But it needs
conflict-zone infrastructure that does not exist, and building that is
larger than fixing the current label.

**B.0 (rename only)** is the fallback if a re-run is not feasible.
It's honest but the physics remains wrong.

## What must happen before any experiment is re-run

1. Decision B is chosen and written into this document as an addendum.
2. The DCPA computation is added to `_frame_ttc_stats` (if B.1) with
   tests that fire the counterexample and expect it to be non-critical.
3. Every `.pt` file under `data/processed/graphs/` is deleted (they
   carry edge features produced before PR #148; some may also have
   been generated with the wrong threshold if any config changed).
4. Every `.json` under `docs/paper/` is deleted or moved to
   `docs/paper/_pre_item16/` (renamed and preserved, not cited).
5. Every figure under `docs/paper/figures/` is regenerated.
6. `manuscript.md` is regenerated via `scripts/build_manuscript.py`.
7. Every number in `results.md` and `conclusion.md` is re-derived from
   the fresh `comparison.json`.

**No number in `docs/paper/` is trustworthy until step 7 completes.**

## What this document does NOT do

- Does not choose between B.0, B.1, B.2. That is a research decision,
  not a code decision.
- Does not change `_frame_ttc_stats`, `compute_ttc_simple`, or any
  other code. The rename in Decision A is a separate PR.
- Does not touch `docs/paper/`. Every number there remains valid as a
  record of the pre-decision experiment; none should be cited in a
  paper that adopts a new definition.
- Does not re-run any experiment.

## Cross-reference

- `docs/audit/feature_label_circularity.md` — item 1, the
  circularity finding that this document refines
- `docs/audit/positive_control.md` — item 2, proving the label is
  learnable
- `src/graf/ssm/ttc.py` — implementation 1
- `src/graf/graph/edges.py` — implementation 2
- `scripts/label_strategies.py` — implementation 3
- PR #148 — the already-in-collision fix referenced above
- `docs/paper/results.md`, `docs/paper/conclusion.md` — the current
  claims that depend on which label definition is chosen

---

## Addendum - B.1 chosen and implemented (2026-11-07)

Decision: B.1 (DCPA-gated TCPA). A pair is critical only if it also
satisfies DCPA < dcpa_threshold (default 1.5 m). Pre-B.1 the label
fired on any pair with closing_rate > 0.5 and TCPA < 1.5 s regardless
of closest-approach distance.

Files touched:
- scripts/label_strategies.py: dcpa_threshold on _frame_ttc_stats,
  _frame_critical_pair_ids, all five label_* functions, apply_strategy
- scripts/evaluate_vntraffic.py: same parameter on
  build_ttc_positive_frames and build_labels; CLI --ttc-dcpa-threshold
- scripts/compare_baselines.py: --ttc-dcpa-threshold threaded through
- tests/test_label_strategies.py: 4 tests covering the counterexample,
  head-on, permissive-threshold restoration, and apply_strategy
- docs/paper/SUPERSEDED.md: new; every pre-B.1 artifact listed

Every number under docs/paper/ was generated under the old formula and
is now stale. SUPERSEDED.md says so. Regeneration is a separate
workflow dispatch.
