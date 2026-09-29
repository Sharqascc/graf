# Replication — AICC22-Custom, sustained r=5, post-purge

**Status:** directional replication of the VNTraffic finding on a second
clip from the same Zenodo record. **Not** a confirmatory result.

## Setup

| parameter | value |
|---|---|
| clip | AICC22-Custom (Zenodo 18195750) |
| tracks | 12 of 21 (>= 20 frames) |
| windows | 97 |
| label | sustained, run_length=5 |
| split | blocked, **5 folds** (10 too thin at 97 windows) |
| purge gap | 5 frames |

Label distribution: **69 positive / 28 negative**, majority **0.711**.
Negatives per fold: **[1, 0, 5, 18, 4]**. Fold 2 has zero negatives, so
its AUC is undefined and effective n is **4 folds**.

## Results

| model | mean AUC | 95% CI | mean accuracy |
|---|---:|---|---:|
| majority | 0.500 | [0.500, 0.500] | 0.709 |
| logreg | 0.611 | [0.291, 0.930] | 0.459 |
| rf | 0.567 | [0.243, 0.870] | 0.491 |
| single_feature | **0.765** | [0.534, 0.950] | 0.701 |

Leakage after purge: **0**.

## Reading — directional, not confirmatory

The direction replicates: cue 0.765 > rf 0.567, gap −0.199 (larger than
VNTraffic's −0.046). But the CIs are wide enough that neither model is
distinguishable from chance on this clip. What the second clip supports
is *consistency of direction*, not independent significance.

The cue is the more clip-stable estimator: 0.803 → 0.765 across clips,
vs the RF's 0.757 → 0.567.
