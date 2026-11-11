# P2 results: TTC vs PET on SCI

**Status:** first results, 2026-11-10. One site. Two metric conventions
stated explicitly in section 5.

## Question

`P1_RESULTS.md` showed window-level TTC labels are opportunity counters.
The next question: at the **pair level**, do the two standard SSMs (TTC,
PET) agree on which pairs are dangerous?

## Method chain

Same SCI pipeline as P1. 34,195 rows, 264 tracks after cleaning. Scale
29.89 px/m (TTC is scale-invariant; PET uses scale for the 2 m zone).

Analysis via `scripts/ttc_pet_comparison.py` (CI-gated, 11 tests):

1. Candidate pairs: any pair within 3 m at some frame -> 815 pairs
2. Per-pair **min TTC** at B.1 collision radius 1.5 m, threshold 1.5 s
3. Per-pair **PET** with proximity 2 m, threshold 5 s
4. Agreement matrix + Jaccard

## Main result

| | count | % of 815 |
|---|---:|---:|
| TTC critical (<= 1.5 s) | 485 | 59.5 |
| PET critical (<= 5.0 s) | 647 | 79.4 |
| **both** | **399** | **49.0** |
| **TTC only** | **86** | **10.6** |
| **PET only** | **248** | **30.4** |
| neither | 82 | 10.1 |

**Jaccard = 0.544.** The two metrics agree on half the pairs and disagree
on 41%.

The asymmetry is the finding: **PET flags 30% more pairs than TTC, and
248 pairs are flagged by PET but not TTC.** Only 86 pairs go the other
way. PET is much more permissive here than TTC.

## Mechanism 1: PET-only (248 pairs)

**What PET catches that TTC misses: pairs already inside the collision
radius.**

Worked example from the data: `(1, 3)` had `dist = 1.39 m`, `TTC = inf`,
`PET = 0.00 s`.

- The two actors are 1.39 m apart, inside TTC's 1.5 m collision radius
- TTC returns `already_in_collision` status, which by this paper's
  convention does **not** count as a critical prediction (see section 5)
- PET: closest-approach point is ~1.39 m, both actors are inside the 2 m
  zone simultaneously, so PET = 0 (zone overlap)

Standard interpretation of this pair: **very close, arguably already in
conflict**. TTC's "no future collision predicted" reading and PET's
"0 s gap" reading disagree because they answer different questions.

The `(1, 3)` case is not rare. Most of the 248 PET-only pairs are this
geometric pattern: real proximity + already-close status.

## Mechanism 2: TTC-only (86 pairs)

**What TTC catches that PET misses: pairs on collision-projecting paths
whose actual zone crossings are far apart in time.**

Worked example: `(4, 47)` had `dist = 2.01 m`, `TTC = 0.24 s`,
`PET = 8.63 s`.

- TTC's constant-velocity quadratic projects a collision in 0.24 s. From
  the metric's perspective, this is severe.
- PET: closest approach 2.01 m, zone radius 2 m (barely grazed). One
  actor exits before the other enters, so PET = 8.63 s, i.e., the two
  trajectories cross the reference point almost 9 seconds apart.
- In reality, these actors almost certainly **did not collide** -- they
  passed 2 m apart while extrapolating trajectories would have predicted
  collision.

This is TTC's known extrapolation weakness. **86 pairs in this data look
more dangerous under constant-velocity projection than under zone-crossing
timing.** A human auditing the video would likely agree with PET (they
didn't collide), but the metric TTC reports "0.24 s to collision."

## Interpretation

TTC and PET measure different physical scenarios on the same data:

| pattern | who flags it | geometric cause |
|---|---|---|
| already close, no future collision projected | **PET** | inside radius, TTC status filtered |
| collision in projection, but trajectories miss | **TTC** | extrapolation artifact |
| actual conflict in both senses | both | genuine near-miss |
| neither | neither | far apart |

**Neither metric is "right" without ground truth.** We can state that
they disagree, quantify the disagreement (Jaccard 0.544), and identify
the mechanism (zone overlap vs extrapolation). We cannot say PET is safer
or TTC is broken.

The 49% overlap (399 pairs) is the pairs where both metrics
independently identify the same scenario. The 41% disagreement is the
interesting part -- it means no single SSM is a complete picture.

## Scope and claim ceiling

- **One site.** SCI, 90 seconds, one intersection.
- **No human labels.** Cannot say which metric is "correct" per pair.
- **Pairwise-zone PET, not standard PET** (section 5 caveat below).
- **TTC excludes `already_in_collision`** from "critical" (section 5).
- **Claim ceiling:** on one heterogeneous-traffic site, TTC and
  pairwise-zone PET flag partially overlapping sets (Jaccard 0.544) with
  two identifiable geometric causes.

## Section 5: the two conventions

**Convention 1: pairwise-zone PET.** Standard PET uses fixed manual
conflict zones per intersection (defined polygons at crossings or merge
points). This implementation instead uses each pair's own closest-approach
point as the zone. This biases PET toward firing on already-close pairs:
if two actors come within 2 m, the zone is *where they were close*, so
PET is small. **Standard PET applied to the same data would likely flag
fewer pairs**, because most pairs never enter a fixed conflict zone at
all. We call this "pairwise-zone PET" to distinguish it.

**Convention 2: TTC excludes `already_in_collision`.** When two actors
are already within the 1.5 m collision radius, TTC returns
`status = already_in_collision` with `ttc_seconds = 0`. This paper's
`compute_pair_metrics` only counts `status = collision_predicted` as
TTC-critical, so pairs already colliding do not enter the TTC-flagged
set. Under the alternative convention (TTC = 0 counts as critical), TTC
would flag more pairs and the Jaccard would rise. That sensitivity is
not reported here; it is a required follow-up before submission.
