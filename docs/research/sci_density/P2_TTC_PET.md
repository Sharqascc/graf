# P2 results: TTC vs PET on SCI

**Status:** first results, 2026-11-10. One site. Two metric conventions stated explicitly in section 5.

## Question

`P1_RESULTS.md` showed window-level TTC labels are opportunity counters. The next question: at the **pair level**, do the two standard SSMs (TTC, PET) agree on which pairs are dangerous?

## Method chain

Same SCI pipeline as P1. 34,195 rows, 264 tracks after cleaning. Scale 29.89 px/m (TTC is scale-invariant; PET uses scale for the 2 m zone radius).

Analysis via `scripts/ttc_pet_comparison.py` (CI-gated, 11 tests):

1. Candidate pairs: any pair within 3 m at some frame -> 815 pairs
2. Per-pair min TTC at B.1 collision radius 1.5 m, threshold 1.5 s
3. Per-pair PET with proximity 2 m, threshold 5 s
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

**Jaccard = 0.544.** The two metrics agree on half the pairs and disagree on 41%. The asymmetry is the finding: PET flags 30% more pairs than TTC, and 248 pairs are flagged by PET but not TTC. Only 86 pairs go the other way.

## Mechanism 1: PET-only (248 pairs)

**What PET catches that TTC misses: pairs already inside the collision radius.**

Worked example from the data: `(1, 3)` had `dist = 1.39 m`, `TTC = inf`, `PET = 0.00 s`.

- Two actors are 1.39 m apart, inside TTC's 1.5 m collision radius
- TTC returns `already_in_collision` status, which by this paper's convention does not count as a critical prediction (see section 5)
- PET: closest-approach point is ~1.39 m, both actors inside the 2 m zone simultaneously, so PET = 0 (zone overlap)

Standard interpretation: **very close, arguably already in conflict**. TTC's "no future collision" reading and PET's "0 s gap" reading disagree because they answer different questions. Most of the 248 PET-only pairs are this geometric pattern.

## Mechanism 2: TTC-only (86 pairs)

**What TTC catches that PET misses: pairs on collision-projecting paths whose actual zone crossings are far apart in time.**

Worked example: `(4, 47)` had `dist = 2.01 m`, `TTC = 0.24 s`, `PET = 8.63 s`.

- TTC's constant-velocity quadratic projects a collision in 0.24 s. Severe from the metric's perspective.
- PET: closest approach 2.01 m, zone radius 2 m (barely grazed). One actor exits before the other enters, so PET = 8.63 s.
- In reality, these actors almost certainly did not collide -- they passed 2 m apart while extrapolating trajectories would have predicted collision.

This is TTC's known extrapolation weakness. **86 pairs look more dangerous under constant-velocity projection than under zone-crossing timing.**

## Interpretation

TTC and PET measure different physical scenarios on the same data:

| pattern | who flags it | geometric cause |
|---|---|---|
| already close, no future collision | PET | inside radius, TTC status filtered |
| collision in projection, trajectories miss | TTC | extrapolation artifact |
| actual conflict in both senses | both | genuine near-miss |
| neither | neither | far apart |

**Neither metric is "right" without ground truth.** We can state that they disagree, quantify the disagreement (Jaccard 0.544), and identify the mechanism. We cannot say PET is safer or TTC is broken.

## Section 5: two conventions

**Pairwise-zone PET.** Standard PET uses fixed manual conflict zones. This implementation uses each pair's own closest-approach point as the zone. Biases PET toward firing on already-close pairs. Standard PET applied to this data would likely flag fewer pairs.

**TTC excludes `already_in_collision`.** TTC returns `ttc_seconds = 0` with status `already_in_collision` when within radius. This paper's `compute_pair_metrics` only counts `collision_predicted` status as TTC-critical. Alternative convention (TTC=0 counts as critical) would raise Jaccard. Sensitivity not reported; required follow-up before submission.

## Scope and claim ceiling

- One site: SCI, 90 seconds, one intersection
- No human labels: cannot say which metric is "correct" per pair
- Claim ceiling: on one heterogeneous-traffic site, TTC and pairwise-zone PET flag partially overlapping sets (Jaccard 0.544) with two identifiable geometric causes

## Reproduce

The script is `scripts/ttc_pet_comparison.py`. Run with --tracks and --output_dir. Source artifact: SCI pipeline run 42638434164, artifact sci-tracks-42638434164 (id 13685821606), regenerated 2026-11-09. Verified against ad-hoc probe: Jaccard delta 0.0003.
