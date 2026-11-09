"""TTC vs PET pairwise comparison on bbox-derived tracks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
from graf.ssm.pet import PETCalculator
from graf.ssm.ttc import compute_ttc_constant_velocity

TTC_THRESHOLD_S = 1.5
PET_THRESHOLD_S = 5.0
DIST_THRESHOLD_M = 3.0
MIN_DISTANCE_M = 1.5


def scan_candidate_pairs(df, dist_threshold=DIST_THRESHOLD_M):
    """Return pairs of tracks that come within dist_threshold at any frame."""
    pairs = set()
    for _, g in df.groupby("frame_idx"):
        arr = g[["track_id", "x_m", "y_m"]].dropna().to_numpy()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(arr[i, 1:3] - arr[j, 1:3]) < dist_threshold:
                    a, b = int(arr[i, 0]), int(arr[j, 0])
                    pairs.add((min(a, b), max(a, b)))
    return pairs


def compute_pair_metrics(
    df, a, b, ttc_threshold=TTC_THRESHOLD_S, pet_threshold=PET_THRESHOLD_S
):
    """Compute TTC and PET for one pair. None if insufficient data."""
    da = df[df["track_id"] == a].sort_values("frame_idx").reset_index(drop=True)
    db = df[df["track_id"] == b].sort_values("frame_idx").reset_index(drop=True)
    if len(da) < 2 or len(db) < 2:
        return None
    # PETCalculator requires t_sec. Derive from frame_idx at 30 fps if
    # the caller did not supply it (t_sec is only used as a difference,
    # so the absolute offset does not matter).
    if "t_sec" not in da.columns:
        da = da.assign(t_sec=da["frame_idx"].astype(float) / 30.0)
    if "t_sec" not in db.columns:
        db = db.assign(t_sec=db["frame_idx"].astype(float) / 30.0)
    common = set(da["frame_idx"]) & set(db["frame_idx"])
    if not common:
        return None
    min_ttc = float("inf")
    min_dist = float("inf")
    for fid in common:
        ra = da[da["frame_idx"] == fid].iloc[0]
        rb = db[db["frame_idx"] == fid].iloc[0]
        if pd.isna(ra["vx"]) or pd.isna(rb["vx"]):
            continue
        p1 = np.array([ra["x_m"], ra["y_m"]])
        v1 = np.array([ra["vx"], ra["vy"]])
        p2 = np.array([rb["x_m"], rb["y_m"]])
        v2 = np.array([rb["vx"], rb["vy"]])
        min_dist = min(min_dist, float(np.linalg.norm(p1 - p2)))
        res = compute_ttc_constant_velocity(p1, v1, p2, v2, min_distance=MIN_DISTANCE_M)
        if res.status == "collision_predicted" and np.isfinite(res.ttc_seconds):
            min_ttc = min(min_ttc, res.ttc_seconds)
    pet_calc = PETCalculator(
        proximity_threshold_m=2.0, critical_threshold_s=pet_threshold
    )
    try:
        ev = pet_calc.compute_pair_pet(da, db, video_id="sci")
        pet = float(ev.min_value) if ev else float("inf")
    except Exception:
        pet = float("inf")
    return {
        "track_a": a,
        "track_b": b,
        "min_ttc": min_ttc,
        "pet": pet,
        "min_dist": min_dist,
        "ttc_critical": min_ttc <= ttc_threshold,
        "pet_critical": pet <= pet_threshold,
    }


def agreement_summary(pair_df):
    """Compute agreement matrix and Jaccard from a pair metrics DataFrame."""
    n = len(pair_df)
    if n == 0:
        return {
            "n_pairs": 0,
            "ttc_critical": 0,
            "pet_critical": 0,
            "both": 0,
            "ttc_only": 0,
            "pet_only": 0,
            "neither": 0,
            "jaccard": 0.0,
        }
    ttc = pair_df["ttc_critical"].astype(bool)
    pet = pair_df["pet_critical"].astype(bool)
    both = int((ttc & pet).sum())
    ttc_only = int((ttc & ~pet).sum())
    pet_only = int((~ttc & pet).sum())
    neither = int((~ttc & ~pet).sum())
    union = both + ttc_only + pet_only
    return {
        "n_pairs": n,
        "ttc_critical": int(ttc.sum()),
        "pet_critical": int(pet.sum()),
        "both": both,
        "ttc_only": ttc_only,
        "pet_only": pet_only,
        "neither": neither,
        "jaccard": float(both / union) if union > 0 else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--output_dir", default="outputs/ttc_pet")
    ap.add_argument("--ttc_threshold", type=float, default=TTC_THRESHOLD_S)
    ap.add_argument("--pet_threshold", type=float, default=PET_THRESHOLD_S)
    ap.add_argument("--dist_threshold", type=float, default=DIST_THRESHOLD_M)
    ap.add_argument("--pixels_per_meter", type=float, default=None)
    a = ap.parse_args(argv)
    from scripts.sci_density_analysis import (
        add_velocities,
        bbox_to_world,
        estimate_pixels_per_meter,
        filter_implausible,
        load_tracks,
    )

    df = load_tracks(a.tracks)
    ppm = a.pixels_per_meter if a.pixels_per_meter else estimate_pixels_per_meter(df)
    print(f"scale: {ppm:.2f} px/m")
    df = bbox_to_world(df, ppm)
    df = add_velocities(df)
    df = filter_implausible(df)
    print(f"df: {len(df)} rows, {df['track_id'].nunique()} tracks")
    pairs = scan_candidate_pairs(df, a.dist_threshold)
    print(f"candidate pairs: {len(pairs)}")
    rows = []
    for idx, (ta, tb) in enumerate(sorted(pairs)):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(pairs)}")
        r = compute_pair_metrics(df, ta, tb, a.ttc_threshold, a.pet_threshold)
        if r is not None:
            rows.append(r)
    rdf = pd.DataFrame(rows)
    print(f"computed: {len(rdf)} pairs")
    summary = agreement_summary(rdf)
    print()
    print("=== agreement matrix ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(out / "pair_metrics.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
